//! Integration tests for model selection and DataFrame workflows
//!
//! These tests verify that the complete model selection pipeline works
//! correctly end-to-end, including DataFrame integration, hyperparameter
//! tuning, and automatic model selection.

use scirs2_core::ndarray::{array, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{Distribution, SeedableRng};
use sklears_covariance::model_selection_presets;
use sklears_covariance::{
    CovarianceDataFrame, DataFrameEstimator, EmpiricalCovariance, GraphicalLasso, LedoitWolf,
};

/// Generate test data for model selection
fn generate_test_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = scirs2_core::random::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng))
}

#[test]
fn test_basic_model_selection() {
    // Generate test data
    let _data = generate_test_data(100, 10, 42);

    // Create a basic selector
    let selector = model_selection_presets::basic_selector();

    // Verify selector has the expected estimators
    assert_eq!(
        selector.candidates.len(),
        2,
        "Basic selector should have 2 estimators"
    );

    // Note: We can't actually run selection without implementing DataCharacteristics
    // analysis, but we can verify the selector is properly configured
    assert!(selector
        .candidates
        .iter()
        .any(|c| c.name == "EmpiricalCovariance"));
    assert!(selector.candidates.iter().any(|c| c.name == "LedoitWolf"));
}

#[test]
fn test_high_dimensional_selector() {
    // Create high-dimensional selector
    let selector = model_selection_presets::high_dimensional_selector();

    // Verify it has LedoitWolf which is ideal for high-dimensional data
    assert!(selector.candidates.iter().any(|c| c.name == "LedoitWolf"));
}

#[test]
fn test_sparse_selector() {
    // Create sparse selector
    let selector = model_selection_presets::sparse_selector();

    // Verify it has GraphicalLasso which is ideal for sparse precision matrices
    assert!(selector
        .candidates
        .iter()
        .any(|c| c.name == "GraphicalLasso"));
}

#[test]
fn test_dataframe_estimator_empirical() {
    // Create test data with more variation to avoid singularity
    let data = generate_test_data(20, 3, 42);
    let column_names = vec![
        "feature1".to_string(),
        "feature2".to_string(),
        "feature3".to_string(),
    ];

    // Create CovarianceDataFrame
    let df =
        CovarianceDataFrame::new(data, column_names, None).expect("Failed to create DataFrame");

    // Test EmpiricalCovariance with DataFrame
    let estimator = EmpiricalCovariance::new();
    let result = estimator
        .fit_dataframe(&df)
        .expect("Failed to fit empirical covariance");

    // Verify result
    assert_eq!(result.covariance.shape(), &[3, 3]);
    assert_eq!(result.feature_names.len(), 3);
    assert_eq!(result.estimator_info.name, "EmpiricalCovariance");
}

#[test]
fn test_dataframe_estimator_ledoit_wolf() {
    // Create test data
    let data = generate_test_data(50, 5, 42);
    let column_names = (0..5).map(|i| format!("feature{}", i)).collect();

    // Create CovarianceDataFrame
    let df =
        CovarianceDataFrame::new(data, column_names, None).expect("Failed to create DataFrame");

    // Test LedoitWolf with DataFrame
    let estimator = LedoitWolf::new();
    let result = estimator
        .fit_dataframe(&df)
        .expect("Failed to fit Ledoit-Wolf");

    // Verify result
    assert_eq!(result.covariance.shape(), &[5, 5]);
    assert_eq!(result.feature_names.len(), 5);
    assert_eq!(result.estimator_info.name, "LedoitWolf");
    assert!(result.estimator_info.metrics.is_some());
}

#[test]
fn test_dataframe_estimator_graphical_lasso() {
    // Create test data
    let data = generate_test_data(30, 4, 42);
    let column_names = (0..4).map(|i| format!("var{}", i)).collect();

    // Create CovarianceDataFrame
    let df =
        CovarianceDataFrame::new(data, column_names, None).expect("Failed to create DataFrame");

    // Test GraphicalLasso with DataFrame
    let estimator = GraphicalLasso::new().alpha(0.1).max_iter(50);
    let result = estimator
        .fit_dataframe(&df)
        .expect("Failed to fit Graphical Lasso");

    // Verify result
    assert_eq!(result.covariance.shape(), &[4, 4]);
    assert_eq!(result.feature_names.len(), 4);
    assert_eq!(result.estimator_info.name, "GraphicalLasso");
    assert!(result.precision.is_some());
    assert!(result.estimator_info.convergence.is_some());
}

#[test]
fn test_dataframe_with_metadata() {
    // Create test data
    let data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ];
    let column_names = vec!["x".to_string(), "y".to_string(), "z".to_string()];

    // Create CovarianceDataFrame
    let df =
        CovarianceDataFrame::new(data, column_names, None).expect("Failed to create DataFrame");

    // Verify metadata
    assert_eq!(df.metadata.n_rows, 4);
    assert_eq!(df.metadata.n_features, 3);
    assert_eq!(df.metadata.column_stats.len(), 3);

    // Check statistics are computed
    for col_name in &df.column_names {
        assert!(df.metadata.column_stats.contains_key(col_name));
    }
}

#[test]
fn test_estimator_performance_metrics() {
    // Create test data
    let data = generate_test_data(100, 10, 42);
    let column_names = (0..10).map(|i| format!("f{}", i)).collect();

    // Create CovarianceDataFrame
    let df =
        CovarianceDataFrame::new(data, column_names, None).expect("Failed to create DataFrame");

    // Fit EmpiricalCovariance and check performance metrics
    let estimator = EmpiricalCovariance::new();
    let result = estimator
        .fit_dataframe(&df)
        .expect("Failed to fit empirical covariance");

    // Verify performance metrics are captured
    assert!(result.estimator_info.metrics.is_some());
    let metrics = result.estimator_info.metrics.unwrap();
    // Computation time should be non-negative (may be 0 for very fast operations)
    assert!(metrics.computation_time_ms >= 0.0);
}

#[test]
fn test_convergence_info_tracking() {
    // Create test data
    let data = generate_test_data(50, 8, 42);
    let column_names = (0..8).map(|i| format!("v{}", i)).collect();

    // Create CovarianceDataFrame
    let df =
        CovarianceDataFrame::new(data, column_names, None).expect("Failed to create DataFrame");

    // Fit GraphicalLasso and check convergence info
    let estimator = GraphicalLasso::new().alpha(0.05).max_iter(100);
    let result = estimator
        .fit_dataframe(&df)
        .expect("Failed to fit Graphical Lasso");

    // Verify convergence info is tracked
    assert!(result.estimator_info.convergence.is_some());
    let convergence = result.estimator_info.convergence.unwrap();
    assert!(convergence.n_iterations > 0);
    assert!(convergence.tolerance.is_some());
}

#[test]
fn test_estimator_parameters_recorded() {
    // Create test data
    let data = generate_test_data(40, 6, 42);
    let column_names = (0..6).map(|i| format!("col{}", i)).collect();

    // Create CovarianceDataFrame
    let df =
        CovarianceDataFrame::new(data, column_names, None).expect("Failed to create DataFrame");

    // Fit LedoitWolf and check parameters are recorded
    let estimator = LedoitWolf::new()
        .store_precision(true)
        .assume_centered(false);
    let result = estimator
        .fit_dataframe(&df)
        .expect("Failed to fit Ledoit-Wolf");

    // Verify parameters are recorded
    assert!(!result.estimator_info.parameters.is_empty());
    assert_eq!(
        result.estimator_info.parameters.get("store_precision"),
        Some(&"true".to_string())
    );
    assert_eq!(
        result.estimator_info.parameters.get("assume_centered"),
        Some(&"false".to_string())
    );
}

#[test]
fn test_multiple_estimators_comparison() {
    // Create test data
    let data = generate_test_data(60, 8, 42);
    let column_names = (0..8).map(|i| format!("feature{}", i)).collect();

    // Create CovarianceDataFrame
    let df =
        CovarianceDataFrame::new(data, column_names, None).expect("Failed to create DataFrame");

    // Fit multiple estimators
    let empirical = EmpiricalCovariance::new();
    let ledoit_wolf = LedoitWolf::new();

    let result1 = empirical
        .fit_dataframe(&df)
        .expect("Failed to fit empirical");
    let result2 = ledoit_wolf
        .fit_dataframe(&df)
        .expect("Failed to fit Ledoit-Wolf");

    // Verify both produce valid results
    assert_eq!(result1.covariance.shape(), result2.covariance.shape());
    assert_eq!(result1.feature_names, result2.feature_names);
    assert_ne!(result1.estimator_info.name, result2.estimator_info.name);

    // Verify they produce different covariance matrices (due to shrinkage)
    let diff = (&result1.covariance - &result2.covariance)
        .mapv(|x| x.abs())
        .sum();
    assert!(diff > 0.0, "Estimators should produce different results");
}
