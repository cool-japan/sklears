//! Simplified integration tests for sklears
//!
//! These tests verify basic cross-crate functionality.

use scirs2_core::ndarray::{array, Array2};
use sklears::metrics::classification::accuracy_score;
use sklears::neighbors::KNeighborsClassifier;
use sklears::prelude::*;
use sklears::traits::PredictProba;
// StandardScaler not available in facade
// use sklears::preprocessing::scaling::StandardScaler;
use sklears::utils::data_generation::make_classification;

#[test]
#[allow(non_snake_case)]
fn test_simple_classification_pipeline() {
    // Generate synthetic data
    let (X, y) = make_classification(50, 4, 3, None, None, 0.0, 1.0, Some(42)).unwrap();

    // Skip preprocessing (StandardScaler not available in facade)
    let X_scaled = X.clone();

    // Train classifier
    let classifier = KNeighborsClassifier::new(3);
    let fitted_classifier = classifier.fit(&X_scaled, &y).unwrap();

    // Make predictions
    let predictions = fitted_classifier.predict(&X_scaled).unwrap();

    // Evaluate performance
    let accuracy = accuracy_score(&y, &predictions).unwrap();

    // Should have reasonable accuracy on training data
    assert!(
        accuracy > 0.6,
        "Accuracy should be reasonable, got {}",
        accuracy
    );

    // Check that predictions have the right shape
    assert_eq!(predictions.len(), y.len());

    // Check that all predicted classes are valid
    for &pred in predictions.iter() {
        assert!(
            (0..=2).contains(&pred),
            "Predicted class {} should be in [0, 2]",
            pred
        );
    }
}

#[test]
#[allow(non_snake_case)]
fn test_knn_probability_predictions() {
    // Generate binary classification data
    let (X, y) = make_classification(30, 3, 2, None, None, 0.0, 1.0, Some(42)).unwrap();

    // Train classifier
    let classifier = KNeighborsClassifier::new(5);
    let fitted_classifier = classifier.fit(&X, &y).unwrap();

    // Test probability predictions
    let probabilities = fitted_classifier.predict_proba(&X).unwrap();

    assert_eq!(probabilities.nrows(), X.nrows());
    assert_eq!(probabilities.ncols(), 2); // Binary classification

    // Probabilities should sum to 1 for each sample
    for i in 0..probabilities.nrows() {
        let row_sum: f64 = probabilities.row(i).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1, got {}",
            row_sum
        );
    }

    // All probabilities should be in [0, 1]
    for &prob in probabilities.iter() {
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability should be in [0, 1], got {}",
            prob
        );
    }
}

#[test]
#[allow(non_snake_case)]
#[ignore = "StandardScaler not available in facade crate"]
fn test_preprocessing_consistency() {
    // Test disabled until StandardScaler is available in facade
}

#[test]
#[allow(non_snake_case)]
fn test_data_generation_functions() {
    // Test make_classification
    let (X, y) = make_classification(20, 3, 2, None, None, 0.0, 1.0, Some(42)).unwrap();

    assert_eq!(X.shape(), &[20, 3]);
    assert_eq!(y.len(), 20);

    // Check that we have the expected number of classes
    let mut classes: Vec<i32> = y.iter().copied().collect();
    classes.sort_unstable();
    classes.dedup();
    assert_eq!(classes.len(), 2);

    // Check reproducibility
    let (X2, y2) = make_classification(20, 3, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    assert_eq!(X, X2);
    assert_eq!(y, y2);
}

#[test]
#[allow(non_snake_case)]
#[ignore = "StandardScaler not available in facade crate"]
fn test_cross_crate_compatibility() {
    // Test that different crates work together seamlessly
    let (X, y) = make_classification(30, 4, 3, None, None, 0.0, 1.0, Some(42)).unwrap();

    // Use preprocessing from one crate - disabled
    let X_scaled = X.clone();

    // Use classifier from another crate
    let classifier = KNeighborsClassifier::new(3);
    let fitted_classifier = classifier.fit(&X_scaled, &y).unwrap();
    let predictions = fitted_classifier.predict(&X_scaled).unwrap();

    // Use metrics from yet another crate
    let accuracy = accuracy_score(&y, &predictions).unwrap();

    // Everything should work together
    assert!(accuracy > 0.6, "Should achieve reasonable accuracy");
    assert_eq!(predictions.len(), y.len());
}

#[test]
#[allow(non_snake_case)]
fn test_error_handling() {
    // Test dimension mismatch errors
    let X_train = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y_train = array![0, 1, 2];
    let X_test = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(); // Wrong shape

    let classifier = KNeighborsClassifier::new(2);
    let fitted_classifier = classifier.fit(&X_train, &y_train).unwrap();

    // This should fail due to dimension mismatch
    let result = fitted_classifier.predict(&X_test);
    assert!(result.is_err(), "Should fail with dimension mismatch");
}
