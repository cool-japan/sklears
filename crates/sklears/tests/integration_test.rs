//! Integration tests for sklears
//!
//! These tests verify that different crates work together correctly.

use scirs2_core::ndarray::{array, Array2};
use sklears::metrics::classification::{accuracy_score, f1_score, precision_score, recall_score};
use sklears::neighbors::KNeighborsClassifier;
use sklears::prelude::*;
// StandardScaler not available in facade, skipping preprocessing test
// use sklears::preprocessing::StandardScaler;
use sklears::utils::data_generation::make_classification;

#[test]
#[allow(non_snake_case)]
fn test_end_to_end_classification_pipeline() {
    // Generate synthetic data
    let (X, y) = make_classification(100, 4, 3, None, None, 0.0, 1.0, Some(42)).unwrap();

    // Skip preprocessing for now (StandardScaler not available in facade)
    // TODO: Re-enable when preprocessing is properly exposed
    let X_scaled = X.clone();

    // Train classifier
    let classifier = KNeighborsClassifier::new(3);
    let fitted_classifier = classifier.fit(&X_scaled, &y).unwrap();

    // Make predictions
    let predictions = fitted_classifier.predict(&X_scaled).unwrap();

    // Evaluate performance
    let accuracy = accuracy_score(&y, &predictions).unwrap();

    // Should have reasonable accuracy on training data
    assert!(accuracy > 0.7, "Accuracy should be > 0.7, got {}", accuracy);

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
fn test_clustering_and_classification() {
    // Generate blob data for clustering
    // Using make_classification instead as make_blobs is not available
    let (X, y_true) = make_classification(60, 2, 2, None, None, 0.0, 1.0, Some(42)).unwrap();

    // Use KNN to classify based on cluster structure
    let classifier = KNeighborsClassifier::new(5);
    let fitted_classifier = classifier.fit(&X, &y_true).unwrap();

    // Predict on the same data (should be very accurate)
    let predictions = fitted_classifier.predict(&X).unwrap();

    let accuracy = accuracy_score(&y_true, &predictions).unwrap();
    assert!(
        accuracy > 0.6,
        "Accuracy on blob data should be reasonable, got {}",
        accuracy
    );
}

#[test]
fn test_metrics_consistency() {
    // Create a simple binary classification case
    let y_true = array![0, 1, 1, 0, 1, 0, 1, 1, 0, 0];
    let y_pred = array![0, 1, 0, 0, 1, 1, 1, 1, 0, 1];

    // Calculate all metrics
    let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
    let precision = precision_score(&y_true, &y_pred, Some(1)).unwrap();
    let recall = recall_score(&y_true, &y_pred, Some(1)).unwrap();
    let f1 = f1_score(&y_true, &y_pred, Some(1)).unwrap();

    // Basic sanity checks
    assert!((0.0..=1.0).contains(&accuracy));
    assert!((0.0..=1.0).contains(&precision));
    assert!((0.0..=1.0).contains(&recall));
    assert!((0.0..=1.0).contains(&f1));

    // F1 should be harmonic mean of precision and recall
    let expected_f1 = 2.0 * precision * recall / (precision + recall);
    assert!((f1 - expected_f1).abs() < 1e-10);
}

#[test]
#[ignore = "StandardScaler not available in facade crate"]
fn test_preprocessing_pipeline() {
    // Test disabled until StandardScaler is properly exposed in sklears facade
    // TODO: Re-enable when sklears::preprocessing::StandardScaler is available
}

#[test]
#[allow(non_snake_case)]
fn test_data_generation_consistency() {
    // Test that data generation functions produce consistent outputs
    let (X1, y1) = make_classification(50, 3, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    let (X2, y2) = make_classification(50, 3, 2, None, None, 0.0, 1.0, Some(42)).unwrap();

    // With same random seed, should produce identical results
    assert_eq!(X1, X2);
    assert_eq!(y1, y2);

    // Check shapes
    assert_eq!(X1.shape(), &[50, 3]);
    assert_eq!(y1.len(), 50);

    // Check that we have the expected number of classes
    let mut classes: Vec<i32> = y1.iter().copied().collect();
    classes.sort_unstable();
    classes.dedup();
    assert_eq!(classes.len(), 2);
}

#[test]
#[allow(non_snake_case)]
fn test_cross_crate_type_compatibility() {
    // Test that types from different crates work together seamlessly
    let labels = array![0, 1, 2, 1, 0, 2];

    // Test basic functionality with existing data
    let classifier = KNeighborsClassifier::new(3);
    let X_test = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let fitted = classifier.fit(&X_test, &labels).unwrap();
    let predictions = fitted.predict(&X_test).unwrap();

    // Should predict same as input for this trivial case
    assert_eq!(predictions.len(), labels.len());
}
