//! Basic functionality tests without BLAS dependencies
//!
//! These tests focus on algorithms that don't require heavy linear algebra.

#![allow(unexpected_cfgs)]

use scirs2_core::ndarray::{array, Array2};

// Test basic KNN functionality
#[cfg(feature = "neighbors")]
#[test]
#[allow(non_snake_case)]
fn test_knn_basic_functionality() {
    use sklears::neighbors::KNeighborsClassifier;
    use sklears::traits::{Fit, Predict};

    // Simple 2D data
    let X = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 5.0, 6.0, 6.0, 7.0, 7.0, 5.0],
    )
    .unwrap();
    let y = array![0, 0, 0, 1, 1, 1];

    let classifier = KNeighborsClassifier::new(3);
    let fitted_classifier = classifier.fit(&X, &y).unwrap();
    let predictions = fitted_classifier.predict(&X).unwrap();

    assert_eq!(predictions.len(), y.len());

    // Should predict perfectly on training data with this simple case
    for (&pred, &true_label) in predictions.iter().zip(y.iter()) {
        assert_eq!(pred, true_label);
    }
}

// Test metrics without BLAS
#[cfg(feature = "metrics")]
#[test]
#[allow(non_snake_case)]
fn test_basic_metrics() {
    use sklears::metrics::classification::accuracy_score;

    let y_true = array![0, 1, 2, 0, 1, 2];
    let y_pred = array![0, 2, 1, 0, 0, 1];

    let accuracy = accuracy_score(&y_true, &y_pred).unwrap();

    // Manual calculation: 2 correct out of 6 = 0.333...
    assert!((accuracy - 0.3333333333333333).abs() < 1e-10);
}

// Test preprocessing without BLAS
#[cfg(feature = "preprocessing")]
#[test]
#[allow(non_snake_case)]
fn test_label_encoding() {
    use sklears::preprocessing::encoding::LabelEncoder;
    use sklears::traits::Transform;

    let labels = vec!["cat", "dog", "bird", "cat", "dog"];

    let encoder = LabelEncoder::new();
    let fitted_encoder = encoder.fit(&labels, &()).unwrap();
    let encoded = fitted_encoder.transform(&labels).unwrap();

    // Should have consistent encoding
    assert_eq!(encoded[0], encoded[3]); // Both "cat"
    assert_eq!(encoded[1], encoded[4]); // Both "dog"

    // All values should be valid integers (usize is always non-negative)
}

// Test data generation
#[cfg(feature = "datasets")]
#[test]
#[allow(non_snake_case)]
fn test_simple_data_generation() {
    use sklears::datasets::make_classification;

    // Arguments: n_samples, n_features, n_informative, n_redundant, n_classes, random_state
    let (X, y) = make_classification(
        10,       // n_samples
        3,        // n_features
        2,        // n_informative
        0,        // n_redundant
        2,        // n_classes
        Some(42), // random_state
    )
    .unwrap();

    assert_eq!(X.shape(), &[10, 3]);
    assert_eq!(y.len(), 10);

    // Check reproducibility
    let (X2, y2) = make_classification(
        10,       // n_samples
        3,        // n_features
        2,        // n_informative
        0,        // n_redundant
        2,        // n_classes
        Some(42), // random_state
    )
    .unwrap();
    assert_eq!(X, X2);
    assert_eq!(y, y2);
}

// Test basic validation functions
#[test]
#[allow(non_snake_case)]
fn test_basic_validation() {
    // Skipping test due to validation API module not being re-exported
    // TODO: Update when validation module is properly re-exported in sklears facade
    // Placeholder - test intentionally empty pending implementation
}

// Test simple array utilities
#[test]
#[allow(non_snake_case)]
fn test_array_utilities() {
    use sklears::utils::array_utils::label_counts;

    let data = array![1, 2, 2, 3, 1, 3, 3];

    let counts = label_counts(&data);
    assert_eq!(counts.len(), 3);
    assert_eq!(counts[&1], 2);
    assert_eq!(counts[&2], 2);
    assert_eq!(counts[&3], 3);
}

// Test tree algorithms if available (these might work without BLAS)
#[cfg(feature = "tree")]
#[test]
#[allow(non_snake_case)]
fn test_basic_decision_tree() {
    use sklears::traits::{Fit, Predict};
    use sklears::tree::DecisionTreeClassifier;

    // Simple linearly separable data
    let X = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let y = array![0.0, 1.0, 1.0, 0.0]; // XOR pattern

    let tree = DecisionTreeClassifier::new();
    let fitted_tree = tree.fit(&X, &y).unwrap();
    let predictions = fitted_tree.predict(&X).unwrap();

    assert_eq!(predictions.len(), y.len());
    // Trees should be able to learn XOR with enough depth
}

// Property-based test for KNN
#[cfg(all(feature = "neighbors", test))]
#[test]
#[allow(non_snake_case)]
fn test_knn_properties() {
    use sklears::neighbors::KNeighborsClassifier;
    use sklears::traits::{Fit, Predict};

    // Property: KNN should always predict the same class for identical points
    let X = Array2::from_shape_vec(
        (4, 2),
        vec![
            1.0, 2.0, 1.0, 2.0, // Duplicate
            3.0, 4.0, 3.0, 4.0, // Duplicate
        ],
    )
    .unwrap();
    let y = array![0, 0, 1, 1];

    let classifier = KNeighborsClassifier::new(1);
    let fitted_classifier = classifier.fit(&X, &y).unwrap();
    let predictions = fitted_classifier.predict(&X).unwrap();

    // Identical points should have identical predictions
    assert_eq!(predictions[0], predictions[1]);
    assert_eq!(predictions[2], predictions[3]);
}
