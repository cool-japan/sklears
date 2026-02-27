//! Integration tests for sklears-calibration
//!
//! These tests verify that different calibration methods work together
//! correctly and that the API is consistent across all methods.

use scirs2_core::ndarray::{concatenate, Array1, Array2, Axis};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_calibration::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::traits::Fit;

/// Generate reproducible test data
fn generate_test_data(n_samples: usize, seed: u64) -> (Array2<f64>, Array1<i32>) {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n_samples, 2), |_| normal.sample(&mut rng));
    let y = Array1::from_shape_fn(n_samples, |i| if i < n_samples / 2 { 0 } else { 1 });

    (x, y)
}

#[test]
fn test_all_basic_methods_train_successfully() {
    let (x, y) = generate_test_data(200, 42);

    let methods = vec![
        CalibrationMethod::Sigmoid,
        CalibrationMethod::Isotonic,
        CalibrationMethod::Temperature,
        CalibrationMethod::Beta,
    ];

    for method in methods {
        let calibrator = CalibratedClassifierCV::new().method(method.clone()).cv(3);

        let result = calibrator.fit(&x, &y);
        assert!(
            result.is_ok(),
            "Method {:?} should train successfully",
            method
        );

        if let Ok(trained) = result {
            assert_eq!(trained.classes().len(), 2, "Should detect 2 classes");
        }
    }
}

#[test]
fn test_multiclass_methods() {
    let (mut x, mut y) = generate_test_data(300, 123);

    // Extend to 3 classes
    let (x_extra, _) = generate_test_data(100, 456);
    let y_extra = Array1::from_elem(100, 2);

    x = concatenate(Axis(0), &[x.view(), x_extra.view()]).unwrap();
    y = concatenate(Axis(0), &[y.view(), y_extra.view()]).unwrap();

    let methods = vec![
        CalibrationMethod::MulticlassTemperature,
        CalibrationMethod::Dirichlet { concentration: 1.0 },
    ];

    for method in methods {
        let calibrator = CalibratedClassifierCV::new().method(method.clone()).cv(2);

        let result = calibrator.fit(&x, &y);
        assert!(
            result.is_ok(),
            "Multiclass method {:?} should train successfully",
            method
        );

        if let Ok(trained) = result {
            assert_eq!(trained.classes().len(), 3, "Should detect 3 classes");
        }
    }
}

#[test]
fn test_builder_pattern_consistency() {
    let (x, y) = generate_test_data(150, 789);

    // Test various builder configurations
    let configs = vec![
        CalibratedClassifierCV::new()
            .method(CalibrationMethod::Sigmoid)
            .cv(2),
        CalibratedClassifierCV::new()
            .method(CalibrationMethod::Isotonic)
            .cv(5)
            .ensemble(true),
        CalibratedClassifierCV::new()
            .method(CalibrationMethod::Temperature)
            .cv(3)
            .ensemble(false),
    ];

    for calibrator in configs {
        let result = calibrator.fit(&x, &y);
        assert!(result.is_ok(), "Builder pattern should work consistently");
    }
}

#[test]
fn test_small_dataset_handling() {
    // Test with very small dataset (edge case)
    let (x, y) = generate_test_data(20, 111);

    let calibrator = CalibratedClassifierCV::new()
        .method(CalibrationMethod::Sigmoid)
        .cv(2); // Use small CV for small dataset

    let result = calibrator.fit(&x, &y);
    assert!(
        result.is_ok(),
        "Should handle small datasets without panicking"
    );
}

#[test]
fn test_deterministic_results() {
    // Test that same seed produces consistent results
    let (x1, y1) = generate_test_data(200, 999);
    let (x2, y2) = generate_test_data(200, 999);

    // Verify test data is identical
    assert_eq!(x1, x2, "Same seed should produce same features");
    assert_eq!(y1, y2, "Same seed should produce same labels");

    // Train calibrators
    let calibrator1 = CalibratedClassifierCV::new()
        .method(CalibrationMethod::Sigmoid)
        .cv(3);

    let calibrator2 = CalibratedClassifierCV::new()
        .method(CalibrationMethod::Sigmoid)
        .cv(3);

    let result1 = calibrator1.fit(&x1, &y1);
    let result2 = calibrator2.fit(&x2, &y2);

    assert!(result1.is_ok() && result2.is_ok());

    // Both should produce 2 classes
    if let (Ok(t1), Ok(t2)) = (result1, result2) {
        assert_eq!(t1.classes().len(), t2.classes().len());
    }
}

#[test]
fn test_cross_validation_variants() {
    let (x, y) = generate_test_data(300, 2024);

    // Test different CV fold counts
    let cv_folds = vec![2, 3, 5, 10];

    for folds in cv_folds {
        if folds > x.nrows() / 20 {
            // Skip if too many folds for dataset size
            continue;
        }

        let calibrator = CalibratedClassifierCV::new()
            .method(CalibrationMethod::Sigmoid)
            .cv(folds);

        let result = calibrator.fit(&x, &y);
        assert!(result.is_ok(), "Should work with {} CV folds", folds);
    }
}

#[test]
fn test_binning_methods_with_different_bins() {
    let (x, y) = generate_test_data(500, 555);

    let bin_configs = vec![5, 10, 20];

    for n_bins in bin_configs {
        let method = CalibrationMethod::HistogramBinning { n_bins };
        let calibrator = CalibratedClassifierCV::new().method(method).cv(3);

        let result = calibrator.fit(&x, &y);
        assert!(
            result.is_ok(),
            "Histogram binning should work with {} bins",
            n_bins
        );
    }
}

#[test]
fn test_error_handling_invalid_input() {
    // Test with mismatched array lengths
    let x = Array2::zeros((100, 2));
    let y = Array1::zeros(50); // Wrong length!

    let calibrator = CalibratedClassifierCV::new()
        .method(CalibrationMethod::Sigmoid)
        .cv(3);

    let result = calibrator.fit(&x, &y);
    assert!(
        result.is_err(),
        "Should return error for mismatched input lengths"
    );
}

#[test]
fn test_single_class_error() {
    // All samples have same label
    let x = Array2::zeros((50, 2));
    let y = Array1::zeros(50); // All class 0

    let calibrator = CalibratedClassifierCV::new()
        .method(CalibrationMethod::Sigmoid)
        .cv(2);

    let result = calibrator.fit(&x, &y);
    assert!(
        result.is_err(),
        "Should return error for single class problem"
    );
}

#[test]
fn test_ensemble_mode() {
    let (x, y) = generate_test_data(200, 777);

    let calibrator_ensemble = CalibratedClassifierCV::new()
        .method(CalibrationMethod::Sigmoid)
        .cv(3)
        .ensemble(true);

    let calibrator_single = CalibratedClassifierCV::new()
        .method(CalibrationMethod::Sigmoid)
        .cv(3)
        .ensemble(false);

    let result_ensemble = calibrator_ensemble.fit(&x, &y);
    let result_single = calibrator_single.fit(&x, &y);

    assert!(result_ensemble.is_ok(), "Ensemble mode should work");
    assert!(result_single.is_ok(), "Non-ensemble mode should work");
}
