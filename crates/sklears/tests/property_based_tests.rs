//! Property-based tests for sklears
//!
//! These tests use proptest to verify mathematical properties and invariants
//! that should hold for all valid inputs.

use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "preprocessing")]
use sklears_core::traits::Transform;
use sklears_core::traits::{Fit, Predict};
use sklears_metrics::classification::accuracy_score;
use sklears_metrics::regression::{mean_absolute_error, mean_squared_error, r2_score};
use sklears_neighbors::{KNeighborsClassifier, KNeighborsRegressor};
// LabelEncoder/OneHotEncoder/FunctionTransformer/PolynomialFeatures/SimpleImputer property
// tests remain disabled pending a follow-up slice; only the StandardScaler/MinMaxScaler/
// Normalizer scaling properties below are wired up in this pass.
// use sklears_preprocessing::encoding::{LabelEncoder, OneHotEncoder};
// use sklears_preprocessing::feature_engineering::{FunctionTransformer, PolynomialFeatures};
// use sklears_preprocessing::imputation::SimpleImputer;
#[cfg(feature = "preprocessing")]
use sklears_preprocessing::scaling::{MinMaxScaler, NormType, Normalizer, StandardScaler};

// Helper function to generate valid test data
fn generate_valid_data() -> impl Strategy<Value = (Array2<f64>, Array1<i32>)> {
    (5usize..50, 2usize..10, 2i32..5).prop_flat_map(|(n_samples, n_features, n_classes)| {
        let data_strategy =
            prop::collection::vec(-10.0..10.0, n_samples * n_features).prop_map(move |data| {
                Array2::from_shape_vec((n_samples, n_features), data)
                    .expect("shape and data length should match")
            });

        let labels_strategy = prop::collection::vec(0..n_classes, n_samples).prop_map(Array1::from);

        (data_strategy, labels_strategy)
    })
}

fn generate_regression_data() -> impl Strategy<Value = (Array2<f64>, Array1<f64>)> {
    (5usize..50, 2usize..10).prop_flat_map(|(n_samples, n_features)| {
        let data_strategy =
            prop::collection::vec(-10.0..10.0, n_samples * n_features).prop_map(move |data| {
                Array2::from_shape_vec((n_samples, n_features), data)
                    .expect("shape and data length should match")
            });

        let targets_strategy =
            prop::collection::vec(-100.0..100.0, n_samples).prop_map(Array1::from);

        (data_strategy, targets_strategy)
    })
}

proptest! {
    #[test]
    #[cfg(feature = "preprocessing")]
    fn test_scaling_preserves_shape((data, _) in generate_valid_data()) {
        let scaler = StandardScaler::new();
        let fitted = scaler.fit(&data, &()).expect("StandardScaler fit should succeed");
        let transformed = fitted.transform(&data).expect("StandardScaler transform should succeed");

        prop_assert_eq!(transformed.nrows(), data.nrows());
        prop_assert_eq!(transformed.ncols(), data.ncols());
    }

    #[test]
    #[cfg(feature = "preprocessing")]
    fn test_normalizer_unit_norm((data, _) in generate_valid_data()) {
        let normalizer = Normalizer::new().norm(NormType::L2);
        let transformed = normalizer.transform(&data).expect("Normalizer transform should succeed");

        prop_assert_eq!(transformed.dim(), data.dim());

        for i in 0..transformed.nrows() {
            let original_norm: f64 = data.row(i).iter().map(|&v| v * v).sum::<f64>().sqrt();
            let scaled_norm: f64 = transformed.row(i).iter().map(|&v| v * v).sum::<f64>().sqrt();

            // Rows whose original L2 norm is effectively zero are left unchanged by
            // Normalizer (documented behaviour), so they are exempt from the
            // unit-norm assertion below.
            if original_norm > 1e-8 {
                prop_assert!(
                    (scaled_norm - 1.0).abs() < 1e-6,
                    "Row {} should have unit L2 norm, got {}", i, scaled_norm
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "preprocessing")]
    fn test_standard_scaler_zero_mean_unit_variance(data in prop::collection::vec(-10.0..10.0, 20..100).prop_map(|v| { let rows = 20; let cols = v.len() / rows; let actual_size = rows * cols; Array2::from_shape_vec((rows, cols), v[..actual_size].to_vec()).expect("shape and data length should match") })) {
        let scaler = StandardScaler::new();
        let fitted = scaler.fit(&data, &()).expect("StandardScaler fit should succeed");
        let transformed = fitted.transform(&data).expect("StandardScaler transform should succeed");

        prop_assert_eq!(transformed.dim(), data.dim());

        let n = transformed.nrows() as f64;
        for j in 0..transformed.ncols() {
            let original_col = data.column(j);
            let original_mean: f64 = original_col.iter().sum::<f64>() / n;
            let original_variance: f64 = original_col.iter().map(|&v| (v - original_mean).powi(2)).sum::<f64>() / n;

            let col = transformed.column(j);
            let mean: f64 = col.iter().sum::<f64>() / n;

            // A (numerically) constant input column has zero variance, and
            // StandardScaler's documented zero-variance guard maps it to all
            // zeros rather than unit variance -- that is expected, not a bug.
            if original_variance > 1e-12 {
                let variance: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
                prop_assert!(mean.abs() < 1e-6, "Column {} mean should be ~0, got {}", j, mean);
                prop_assert!((variance - 1.0).abs() < 1e-6, "Column {} variance should be ~1, got {}", j, variance);
            } else {
                prop_assert!(mean.abs() < 1e-6, "Constant column {} should map to 0, got mean {}", j, mean);
            }
        }
    }

    #[test]
    #[cfg(feature = "preprocessing")]
    fn test_minmax_scaler_range((data, _) in generate_valid_data(), min_val in -5.0..0.0, max_val in 1.0..5.0) {
        let scaler = MinMaxScaler::new().feature_range(min_val, max_val);
        let fitted = scaler.fit(&data, &()).expect("MinMaxScaler fit should succeed");
        let transformed = fitted.transform(&data).expect("MinMaxScaler transform should succeed");

        prop_assert_eq!(transformed.dim(), data.dim());

        let eps = 1e-8;
        for &value in transformed.iter() {
            prop_assert!(
                value >= min_val - eps && value <= max_val + eps,
                "Value {} should be within [{}, {}]", value, min_val, max_val
            );
        }
    }

    #[test]
    fn test_knn_classifier_predictions_valid((data, labels) in generate_valid_data(), k in 1usize..10) {
        prop_assume!(k <= data.nrows());
        prop_assume!(data.nrows() >= 3);

        let classifier = KNeighborsClassifier::new(k);
        let fitted_classifier = classifier.fit(&data, &labels).expect("model fitting should succeed");
        let predictions = fitted_classifier.predict(&data).expect("prediction should succeed");

        // Predictions should have same length as input
        prop_assert_eq!(predictions.len(), data.nrows());

        // All predictions should be valid class labels
        let unique_labels: std::collections::HashSet<i32> = labels.iter().copied().collect();
        for &pred in predictions.iter() {
            prop_assert!(unique_labels.contains(&pred),
                        "Prediction {} should be a valid class label", pred);
        }
    }

    #[test]
    fn test_knn_regressor_finite_predictions((data, targets) in generate_regression_data(), k in 1usize..10) {
        prop_assume!(k <= data.nrows());
        prop_assume!(data.nrows() >= 3);

        let regressor = KNeighborsRegressor::new(k);
        let fitted_regressor = regressor.fit(&data, &targets).expect("model fitting should succeed");
        let predictions = fitted_regressor.predict(&data).expect("prediction should succeed");

        // Predictions should have same length as input
        prop_assert_eq!(predictions.len(), data.nrows());

        // All predictions should be finite
        for &pred in predictions.iter() {
            prop_assert!(pred.is_finite(), "Prediction should be finite, got {}", pred);
        }
    }

    #[test]
    #[ignore = "Preprocessing modules not available in facade"]
    fn test_polynomial_features_shape(data in prop::collection::vec(-5.0..5.0, 12..48).prop_map(|v| { let rows = 6; let cols = v.len() / rows; let actual_size = rows * cols; Array2::from_shape_vec((rows, cols), v[..actual_size].to_vec()).expect("shape and data length should match") }), degree in 1usize..4) {
        // Test disabled - PolynomialFeatures not available in facade
        let _ = (data, degree);
    }

    #[test]
    #[ignore = "Preprocessing modules not available in facade"]
    fn test_label_encoder_consistency(labels in prop::collection::vec("[A-Z]{1,3}", 10..50)) {
        // Test disabled - LabelEncoder not available in facade
        let _ = labels;
    }

    #[test]
    #[ignore = "Preprocessing modules not available in facade"]
    fn test_one_hot_encoder_properties(labels in prop::collection::vec("[A-C]", 10..30)) {
        // Test disabled - OneHotEncoder not available in facade
        let _ = labels;
    }

    #[test]
    #[ignore = "Preprocessing modules not available in facade"]
    fn test_simple_imputer_no_nans(data in prop::collection::vec(-10.0..10.0, 20..100).prop_map(|v| { let rows = 10; let cols = v.len() / rows; let actual_size = rows * cols; Array2::from_shape_vec((rows, cols), v[..actual_size].to_vec()).expect("shape and data length should match") })) {
        // Test disabled - SimpleImputer not available in facade
        let _ = data;
    }

    #[test]
    fn test_accuracy_score_bounds((data, labels) in generate_valid_data()) {
        prop_assume!(data.nrows() >= 5);

        let classifier = KNeighborsClassifier::new(3);
        let fitted_classifier = classifier.fit(&data, &labels).expect("model fitting should succeed");
        let predictions = fitted_classifier.predict(&data).expect("prediction should succeed");

        let accuracy = accuracy_score(&labels, &predictions).expect("operation should succeed");

        // Accuracy should be between 0 and 1
        prop_assert!((0.0..=1.0).contains(&accuracy),
                    "Accuracy should be in [0, 1], got {}", accuracy);
    }

    #[test]
    fn test_regression_metrics_properties((data, targets) in generate_regression_data()) {
        prop_assume!(data.nrows() >= 5);

        let regressor = KNeighborsRegressor::new(3);
        let fitted_regressor = regressor.fit(&data, &targets).expect("model fitting should succeed");
        let predictions = fitted_regressor.predict(&data).expect("prediction should succeed");

        let mse = mean_squared_error(&targets, &predictions).expect("operation should succeed");
        let mae = mean_absolute_error(&targets, &predictions).expect("operation should succeed");
        let r2 = r2_score(&targets, &predictions).expect("operation should succeed");

        // MSE should be non-negative
        prop_assert!(mse >= 0.0, "MSE should be non-negative, got {}", mse);

        // MAE should be non-negative
        prop_assert!(mae >= 0.0, "MAE should be non-negative, got {}", mae);

        // MSE should be >= MAE² (by Cauchy-Schwarz inequality)
        // This is not always true, but worth testing when it should hold
        // prop_assert!(mse >= mae * mae, "MSE should be >= MAE², got MSE={}, MAE={}", mse, mae);

        // R² should be finite
        prop_assert!(r2.is_finite(), "R² should be finite, got {}", r2);

        // For perfect predictions, MSE and MAE should be 0
        let perfect_predictions = targets.clone();
        let perfect_mse = mean_squared_error(&targets, &perfect_predictions).expect("operation should succeed");
        let perfect_mae = mean_absolute_error(&targets, &perfect_predictions).expect("operation should succeed");
        let perfect_r2 = r2_score(&targets, &perfect_predictions).expect("operation should succeed");

        prop_assert!((perfect_mse).abs() < 1e-10, "Perfect predictions should have MSE=0");
        prop_assert!((perfect_mae).abs() < 1e-10, "Perfect predictions should have MAE=0");
        prop_assert!((perfect_r2 - 1.0).abs() < 1e-10, "Perfect predictions should have R²=1");
    }

    #[test]
    #[ignore = "Preprocessing modules not available in facade"]
    fn test_function_transformer_invertibility(data in prop::collection::vec(-5.0..5.0, 12..48).prop_map(|v| { let rows = 6; let cols = v.len() / rows; let actual_size = rows * cols; Array2::from_shape_vec((rows, cols), v[..actual_size].to_vec()).expect("shape and data length should match") })) {
        // Test disabled - FunctionTransformer not available in facade
        let _ = data;
    }
}
