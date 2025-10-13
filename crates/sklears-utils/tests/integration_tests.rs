//! Integration tests for sklears-utils
//!
//! These tests verify that different utility modules work correctly together
//! in realistic machine learning workflows.

use scirs2_core::ndarray::Array1;
use sklears_utils::{
    array_utils::{array_standardize, safe_indexing_2d},
    data_generation::{make_classification, make_regression},
    metrics::{euclidean_distance, manhattan_distance},
    parallel::ParallelIterator,
    preprocessing::{FeatureScaler, OutlierDetector},
    random::{set_random_state, train_test_split_indices},
    validation::check_array_2d,
};

#[test]
fn test_end_to_end_classification_workflow() {
    // Set reproducible random state
    set_random_state(42);

    // Generate classification dataset
    let (X, y) = make_classification(1000, 20, 5, None, None, 0.1, 1.0, Some(42)).unwrap();

    // Validate the generated data
    assert!(check_array_2d(&X).is_ok());

    // Split into train/test
    let (train_indices, _test_indices) =
        train_test_split_indices(X.nrows(), 0.2, true, Some(42)).unwrap();
    let X_train = safe_indexing_2d(&X, &train_indices).unwrap();

    // Preprocess the training data
    let (X_train_scaled, _means, _stds) = FeatureScaler::standard_scale(&X_train).unwrap();

    // Test distance calculations between samples
    for i in 0..std::cmp::min(10, X_train_scaled.nrows()) {
        for j in (i + 1)..std::cmp::min(10, X_train_scaled.nrows()) {
            let row1 = X_train_scaled.row(i).to_owned();
            let row2 = X_train_scaled.row(j).to_owned();

            let euclidean_dist = euclidean_distance(&row1, &row2);
            let manhattan_dist = manhattan_distance(&row1, &row2);

            assert!(euclidean_dist >= 0.0);
            assert!(manhattan_dist >= 0.0);
        }
    }

    println!("End-to-end classification workflow completed successfully");
}

#[test]
fn test_end_to_end_regression_workflow() {
    // Set reproducible random state
    set_random_state(123);

    // Generate regression dataset
    let (X, y) = make_regression(500, 15, None, 0.1, 1.0, Some(42)).unwrap();

    // Validate the generated data
    assert!(check_array_2d(&X).is_ok());

    // Split into train/test
    let (train_indices, _test_indices) =
        train_test_split_indices(X.nrows(), 0.3, true, Some(123)).unwrap();
    let X_train = safe_indexing_2d(&X, &train_indices).unwrap();
    let y_train = Array1::from_vec(train_indices.iter().map(|&i| y[i]).collect::<Vec<_>>());

    // Feature scaling
    let (X_train_scaled, means, stds) = FeatureScaler::standard_scale(&X_train).unwrap();
    assert_eq!(means.len(), X_train.ncols());
    assert_eq!(stds.len(), X_train.ncols());

    // Verify standardization worked correctly
    for col_idx in 0..X_train_scaled.ncols() {
        let column = X_train_scaled.column(col_idx);
        let mean = column.sum() / column.len() as f64;
        let variance = column.map(|&x| (x - mean).powi(2)).sum() / column.len() as f64;

        assert!(
            (mean.abs()) < 1e-10,
            "Mean should be close to 0 after standardization"
        );
        assert!(
            (variance - 1.0).abs() < 1e-10,
            "Variance should be close to 1 after standardization"
        );
    }

    // Test outlier detection on target variable
    let outlier_indices = OutlierDetector::iqr_outliers(&y_train.view(), 1.5);
    assert!(outlier_indices.len() < y_train.len() / 5); // Expect less than 20% outliers

    println!("End-to-end regression workflow completed successfully");
}

#[test]
fn test_parallel_processing_integration() {
    set_random_state(456);

    // Generate a larger dataset for parallel processing
    let (X, _y) = make_classification(2000, 50, 10, None, None, 0.05, 1.0, Some(456)).unwrap();

    // Use parallel iterator for distance calculations
    let sample_indices: Vec<usize> = (0..std::cmp::min(100, X.nrows())).collect();

    let parallel_iter = ParallelIterator::new(sample_indices.clone());
    let distances: Vec<f64> = parallel_iter
        .map(move |i| {
            let row = X.row(i).to_owned();
            let first_row = X.row(0).to_owned();
            euclidean_distance(&row, &first_row)
        })
        .unwrap();

    assert_eq!(distances.len(), sample_indices.len());
    assert_eq!(distances[0], 0.0); // Distance from first row to itself should be 0

    // Verify all distances are non-negative
    for &dist in &distances {
        assert!(dist >= 0.0);
    }

    println!("Parallel processing integration test completed successfully");
}

#[test]
fn test_data_validation_pipeline() {
    set_random_state(789);

    // Generate test data
    let (X, _y) = make_classification(300, 10, 3, None, None, 0.0, 1.0, Some(789)).unwrap();

    // Test array validation functions
    assert!(check_array_2d(&X).is_ok());

    println!("Data validation pipeline test completed successfully");
}

#[test]
fn test_cross_module_data_flow() {
    set_random_state(999);

    // Test data flowing through multiple modules
    let (X, y) = make_regression(400, 8, None, 0.1, 1.0, Some(999)).unwrap();

    // 1. Data generation -> Array utilities
    let original_stats = array_standardize(&X.column(0).to_owned()).unwrap();
    // Check that the standardized data has reasonable properties
    let mean = original_stats.sum() / original_stats.len() as f64;
    assert!(
        mean.abs() < 1e-8,
        "Mean should be close to 0 after standardization"
    );

    // 2. Array utilities -> Preprocessing
    let (X_scaled, _means, _stds) = FeatureScaler::standard_scale(&X).unwrap();

    // 3. Preprocessing -> Metrics
    let sample1 = X_scaled.row(0).to_owned();
    let sample2 = X_scaled.row(1).to_owned();
    let distance = euclidean_distance(&sample1, &sample2);
    assert!(distance >= 0.0);

    // 4. Validation -> Random sampling
    let (train_indices, test_indices) =
        train_test_split_indices(X.nrows(), 0.2, true, Some(999)).unwrap();
    assert!(train_indices.len() > 0);
    assert!(test_indices.len() > 0);

    println!("Cross-module data flow test completed successfully");
}

#[test]
fn test_performance_integration() {
    use std::time::Instant;

    set_random_state(2024);

    // Test performance of large-scale operations
    let start = Instant::now();
    let (X, _y) = make_classification(10000, 100, 10, None, None, 0.0, 1.0, Some(2024)).unwrap();
    let data_generation_time = start.elapsed();

    let start = Instant::now();
    let (X_scaled, _means, _stds) = FeatureScaler::standard_scale(&X).unwrap();
    let scaling_time = start.elapsed();

    let start = Instant::now();
    let sample1 = X_scaled.row(0).to_owned();
    let sample2 = X_scaled.row(1).to_owned();
    let _distance = euclidean_distance(&sample1, &sample2);
    let distance_time = start.elapsed();

    // Verify performance is reasonable (these are lenient bounds for CI)
    assert!(
        data_generation_time.as_millis() < 5000,
        "Data generation took too long"
    );
    assert!(scaling_time.as_millis() < 1000, "Scaling took too long");
    assert!(
        distance_time.as_micros() < 1000,
        "Distance computation took too long"
    );

    println!("Performance integration test completed successfully");
    println!("  Data generation: {:?}", data_generation_time);
    println!("  Scaling: {:?}", scaling_time);
    println!("  Distance: {:?}", distance_time);
}
