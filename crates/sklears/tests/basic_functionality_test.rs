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
    .expect("operation should succeed");
    let y = array![0, 0, 0, 1, 1, 1];

    let classifier = KNeighborsClassifier::new(3);
    let fitted_classifier = classifier
        .fit(&X, &y)
        .expect("model fitting should succeed");
    let predictions = fitted_classifier
        .predict(&X)
        .expect("prediction should succeed");

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

    let accuracy = accuracy_score(&y_true, &y_pred).expect("operation should succeed");

    // Manual calculation: 2 correct out of 6 = 0.333...
    assert!((accuracy - 0.3333333333333333).abs() < 1e-10);
}

// Test preprocessing without BLAS
#[cfg(feature = "preprocessing")]
#[test]
#[allow(non_snake_case)]
fn test_label_encoding() {
    use sklears::preprocessing::encoding::LabelEncoder;

    let labels = vec!["cat", "dog", "bird", "cat", "dog"];

    let mut encoder = LabelEncoder::new();
    let encoded = encoder
        .fit_transform(&labels)
        .expect("label encoder fit_transform should succeed");

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
    .expect("operation should succeed");

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
    .expect("operation should succeed");
    assert_eq!(X, X2);
    assert_eq!(y, y2);
}

// Test basic validation functions
#[test]
#[allow(non_snake_case)]
fn test_basic_validation() {
    use sklears::error::validate::check_consistent_length;
    use sklears::types::arrays::validation::{check_classification_targets, check_finite};
    use sklears::validation::ml::{validate_learning_rate, validate_n_clusters};

    let X = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("shape and data length should match");

    // check_consistent_length: matching sample counts pass, mismatched counts fail
    let y_ok = array![0.0, 1.0, 1.0, 0.0];
    assert!(check_consistent_length(&X, &y_ok).is_ok());
    let y_bad = array![0.0, 1.0];
    assert!(check_consistent_length(&X, &y_bad).is_err());

    // check_finite: arrays without NaN/Inf pass, arrays containing NaN fail
    assert!(check_finite(&X).is_ok());
    let X_with_nan = Array2::from_shape_vec((2, 2), vec![0.0, f64::NAN, 1.0, 2.0])
        .expect("shape and data length should match");
    assert!(check_finite(&X_with_nan).is_err());

    // check_classification_targets: non-negative integer targets pass, negative fail
    let targets_ok = array![0, 1, 2, 1, 0];
    assert!(check_classification_targets(&targets_ok).is_ok());
    let targets_bad = array![0, -1, 2];
    assert!(check_classification_targets(&targets_bad).is_err());

    // validate_learning_rate: must be positive and finite
    assert!(validate_learning_rate(0.01).is_ok());
    assert!(validate_learning_rate(0.0).is_err());
    assert!(validate_learning_rate(-1.0).is_err());

    // validate_n_clusters: must be positive and not exceed the sample count
    assert!(validate_n_clusters(2, 10).is_ok());
    assert!(validate_n_clusters(0, 10).is_err());
    assert!(validate_n_clusters(11, 10).is_err());
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
    let X = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("shape and data length should match");
    let y = array![0, 1, 1, 0]; // XOR pattern

    let tree = DecisionTreeClassifier::new();
    let fitted_tree = tree.fit(&X, &y).expect("model fitting should succeed");
    let predictions = fitted_tree.predict(&X).expect("prediction should succeed");

    assert_eq!(predictions.len(), y.len());
    // Note: this 4-corner XOR pattern is a classic adversarial case for greedy,
    // single-feature-split CART trees: every candidate root split (feature 0
    // or feature 1 at any threshold) yields exactly a 50/50 class split on
    // both sides, i.e. zero impurity decrease, so a standard greedy tree never
    // splits at all here and just predicts the majority class regardless of
    // `max_depth`. This is therefore a smoke test (fit/predict run and return
    // correctly-shaped output), not a correctness test; see
    // `sklears_tree::classifier::tests::test_classifier_learns_separable_blobs`
    // for an accuracy-based non-degeneracy check on data that greedy trees
    // can actually separate.
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
    .expect("operation should succeed");
    let y = array![0, 0, 1, 1];

    let classifier = KNeighborsClassifier::new(1);
    let fitted_classifier = classifier
        .fit(&X, &y)
        .expect("model fitting should succeed");
    let predictions = fitted_classifier
        .predict(&X)
        .expect("prediction should succeed");

    // Identical points should have identical predictions
    assert_eq!(predictions[0], predictions[1]);
    assert_eq!(predictions[2], predictions[3]);
}
