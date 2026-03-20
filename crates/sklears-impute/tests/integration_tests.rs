//! Comprehensive integration tests for sklears-impute
//!
//! These tests verify end-to-end functionality including:
//! - Complete imputation workflows
//! - Pipeline compositions
//! - Cross-method consistency
//! - Real-world scenarios

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Array2};
use scirs2_core::random::thread_rng;
use sklears_core::traits::{Fit, Transform};
use sklears_impute::{
    analyze_missing_patterns, missing_completeness_matrix, missing_correlation_matrix, KNNImputer,
    MissingIndicator, SimpleImputer,
};

/// Generate test data with known missing patterns
fn generate_test_data() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, f64::NAN, 7.0, 8.0],
        [9.0, 10.0, f64::NAN, 12.0],
        [13.0, 14.0, 15.0, f64::NAN],
        [f64::NAN, 18.0, 19.0, 20.0],
    ]
}

/// Test complete imputation pipeline with simple imputer
#[test]
fn test_simple_imputation_pipeline() {
    let data = generate_test_data();

    // Test mean strategy
    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let fitted = imputer.fit(&data.view(), &()).expect("Fit failed");
    let result = fitted.transform(&data.view()).expect("Transform failed");

    // Verify no NaN values remain
    assert!(
        result.iter().all(|&x| !x.is_nan()),
        "NaN values remain after imputation"
    );

    // Verify shape is preserved
    assert_eq!(result.dim(), data.dim(), "Shape changed after imputation");

    // Verify non-missing values are unchanged
    assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[0, 1]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[0, 2]], 3.0, epsilon = 1e-10);
}

/// Test KNN imputation pipeline
#[test]
fn test_knn_imputation_pipeline() {
    let data = generate_test_data();

    let imputer = KNNImputer::new().n_neighbors(2);
    let fitted = imputer.fit(&data.view(), &()).expect("Fit failed");
    let result = fitted.transform(&data.view()).expect("Transform failed");

    // Verify no NaN values remain
    assert!(
        result.iter().all(|&x| !x.is_nan()),
        "NaN values remain after KNN imputation"
    );

    // Verify shape is preserved
    assert_eq!(result.dim(), data.dim(), "Shape changed after imputation");

    // Verify non-missing values are unchanged
    assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
}

/// Test multiple imputation methods on same data
#[test]
fn test_multiple_methods_consistency() {
    let data = generate_test_data();

    // Simple mean imputation
    let simple_mean = SimpleImputer::new().strategy("mean".to_string());
    let fitted_mean = simple_mean.fit(&data.view(), &()).expect("Mean fit failed");
    let result_mean = fitted_mean
        .transform(&data.view())
        .expect("Mean transform failed");

    // Simple median imputation
    let simple_median = SimpleImputer::new().strategy("median".to_string());
    let fitted_median = simple_median
        .fit(&data.view(), &())
        .expect("Median fit failed");
    let result_median = fitted_median
        .transform(&data.view())
        .expect("Median transform failed");

    // Both should have no NaN values
    assert!(result_mean.iter().all(|&x| !x.is_nan()));
    assert!(result_median.iter().all(|&x| !x.is_nan()));

    // Non-missing values should be identical across methods
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            if !data[[i, j]].is_nan() {
                assert_abs_diff_eq!(result_mean[[i, j]], data[[i, j]], epsilon = 1e-10);
                assert_abs_diff_eq!(result_median[[i, j]], data[[i, j]], epsilon = 1e-10);
            }
        }
    }
}

/// Test missing indicator functionality
#[test]
fn test_missing_indicator() {
    let data = generate_test_data();

    let indicator = MissingIndicator::new();
    let fitted = indicator.fit(&data.view(), &()).expect("Fit failed");
    let result = fitted.transform(&data.view()).expect("Transform failed");

    // Result should have same number of rows, columns equal to number of features with missing
    assert_eq!(result.nrows(), data.nrows());

    // Verify indicators are binary (0 or 1)
    for val in result.iter() {
        assert!(
            *val == 0.0 || *val == 1.0,
            "Non-binary indicator value: {}",
            val
        );
    }
}

/// Test missing pattern analysis
#[test]
fn test_missing_pattern_analysis() {
    let data = generate_test_data();

    let patterns = analyze_missing_patterns(&data.view(), f64::NAN).expect("Analysis failed");

    // Should identify distinct patterns
    assert!(!patterns.is_empty(), "No patterns detected");

    // Verify all samples are accounted for
    let total_samples: usize = patterns.values().map(|v| v.len()).sum();
    assert_eq!(
        total_samples,
        data.nrows(),
        "Not all samples accounted for in patterns"
    );
}

/// Test correlation matrix computation
#[test]
fn test_correlation_matrix() {
    let data = generate_test_data();

    let corr_matrix =
        missing_correlation_matrix(&data.view(), f64::NAN).expect("Correlation failed");

    // Should be square matrix with size equal to number of features
    assert_eq!(corr_matrix.nrows(), data.ncols());
    assert_eq!(corr_matrix.ncols(), data.ncols());

    // Diagonal should be 1.0 (or NaN if all values are present/missing)
    for i in 0..corr_matrix.nrows() {
        let val = corr_matrix[[i, i]];
        assert!(
            val == 1.0 || val.is_nan(),
            "Unexpected diagonal value: {}",
            val
        );
    }

    // Should be symmetric
    for i in 0..corr_matrix.nrows() {
        for j in 0..corr_matrix.ncols() {
            if corr_matrix[[i, j]].is_finite() && corr_matrix[[j, i]].is_finite() {
                assert_abs_diff_eq!(corr_matrix[[i, j]], corr_matrix[[j, i]], epsilon = 1e-10);
            }
        }
    }
}

/// Test completeness matrix computation
#[test]
fn test_completeness_matrix() {
    let data = generate_test_data();

    let comp_matrix =
        missing_completeness_matrix(&data.view(), f64::NAN).expect("Completeness failed");

    // Should be square matrix
    assert_eq!(comp_matrix.nrows(), data.ncols());
    assert_eq!(comp_matrix.ncols(), data.ncols());

    // All values should be between 0 and 1
    for val in comp_matrix.iter() {
        assert!(
            *val >= 0.0 && *val <= 1.0,
            "Completeness value out of range: {}",
            val
        );
    }

    // Diagonal should be equal (same feature compared to itself)
    let diag_val = comp_matrix[[0, 0]];
    for i in 0..comp_matrix.nrows() {
        assert_abs_diff_eq!(comp_matrix[[i, i]], diag_val, epsilon = 1e-10);
    }
}

/// Test imputation with all missing column
/// Note: When all values in a column are missing, imputation is not possible
/// and an error should be returned.
#[test]
fn test_all_missing_column() {
    let mut data = generate_test_data();
    // Make one column all NaN
    for i in 0..data.nrows() {
        data[[i, 1]] = f64::NAN;
    }

    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let result = imputer.fit(&data.view(), &());

    // Should return an error when all values in a column are missing
    assert!(
        result.is_err(),
        "Expected error when all values in a column are missing"
    );
}

/// Test imputation with no missing values
#[test]
fn test_no_missing_values() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let fitted = imputer
        .fit(&data.view(), &())
        .expect("Fit with no missing failed");
    let result = fitted
        .transform(&data.view())
        .expect("Transform with no missing failed");

    // Result should be identical to input
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            assert_abs_diff_eq!(result[[i, j]], data[[i, j]], epsilon = 1e-10);
        }
    }
}

/// Test imputation preserves data distribution
#[test]
fn test_distribution_preservation() {
    let mut rng = thread_rng();
    let n_samples = 1000;
    let n_features = 5;

    // Generate data from normal distribution
    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.gen_range(-10.0..10.0);
        }
    }

    // Introduce random missing values (10%)
    for i in 0..n_samples {
        for j in 0..n_features {
            if rng.random::<f64>() < 0.1 {
                data[[i, j]] = f64::NAN;
            }
        }
    }

    // Compute original mean (excluding NaN)
    let mut original_means = vec![0.0; n_features];
    for j in 0..n_features {
        let col: Vec<f64> = (0..n_samples)
            .map(|i| data[[i, j]])
            .filter(|x| !x.is_nan())
            .collect();
        original_means[j] = col.iter().sum::<f64>() / col.len() as f64;
    }

    // Impute with mean
    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let fitted = imputer.fit(&data.view(), &()).expect("Fit failed");
    let result = fitted.transform(&data.view()).expect("Transform failed");

    // Compute imputed means
    let mut imputed_means = vec![0.0; n_features];
    for j in 0..n_features {
        let col: Vec<f64> = (0..n_samples).map(|i| result[[i, j]]).collect();
        imputed_means[j] = col.iter().sum::<f64>() / col.len() as f64;
    }

    // Means should be very close
    for j in 0..n_features {
        assert_abs_diff_eq!(imputed_means[j], original_means[j], epsilon = 0.5);
    }
}

/// Test large dataset performance
#[test]
fn test_large_dataset() {
    let mut rng = thread_rng();
    let n_samples = 10000;
    let n_features = 50;

    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.gen_range(-100.0..100.0);
            if rng.random::<f64>() < 0.15 {
                data[[i, j]] = f64::NAN;
            }
        }
    }

    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let fitted = imputer
        .fit(&data.view(), &())
        .expect("Large dataset fit failed");
    let result = fitted
        .transform(&data.view())
        .expect("Large dataset transform failed");

    assert_eq!(result.dim(), data.dim());
    assert!(result.iter().all(|&x| !x.is_nan()));
}

/// Test edge case: single row
/// Note: With only one row, missing values cannot be imputed from statistics
/// since there's no other data to compute mean/median from.
#[test]
fn test_single_row() {
    let data = array![[1.0, f64::NAN, 3.0, f64::NAN]];

    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let result = imputer.fit(&data.view(), &());

    // Should return an error for single row with missing values
    assert!(
        result.is_err(),
        "Expected error when fitting single row with missing values"
    );
}

/// Test edge case: single column
#[test]
fn test_single_column() {
    let data = array![[1.0], [f64::NAN], [3.0], [f64::NAN], [5.0]];

    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let fitted = imputer
        .fit(&data.view(), &())
        .expect("Single column fit failed");
    let result = fitted
        .transform(&data.view())
        .expect("Single column transform failed");

    assert_eq!(result.ncols(), 1);
    assert!(result.iter().all(|&x| !x.is_nan()));

    // Mean should be (1 + 3 + 5) / 3 = 3
    let expected_mean = 3.0;
    for i in 0..result.nrows() {
        if data[[i, 0]].is_nan() {
            assert_abs_diff_eq!(result[[i, 0]], expected_mean, epsilon = 1e-10);
        }
    }
}

/// Test transform on different data than fit
#[test]
fn test_transform_different_data() {
    let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let test_data = array![[7.0, f64::NAN], [f64::NAN, 10.0], [11.0, 12.0]];

    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let fitted = imputer
        .fit(&train_data.view(), &())
        .expect("Fit on train data failed");
    let result = fitted
        .transform(&test_data.view())
        .expect("Transform on test data failed");

    // Should use statistics from training data
    assert!(result.iter().all(|&x| !x.is_nan()));

    // Non-missing values should be preserved
    assert_abs_diff_eq!(result[[0, 0]], 7.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[2, 0]], 11.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[2, 1]], 12.0, epsilon = 1e-10);
}
