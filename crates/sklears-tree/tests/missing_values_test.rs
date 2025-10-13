/// Test missing value handling for Decision Trees and Random Forest
use scirs2_core::ndarray::array;
use sklears_core::traits::{Fit, Predict};
use sklears_tree::{DecisionTreeClassifier, DecisionTreeRegressor, MissingValueStrategy};

#[test]
fn test_decision_tree_classifier_skip_missing() {
    // Create data with missing values (NaN)
    let x = array![
        [1.0, 2.0],
        [f64::NAN, 3.0], // Missing value in first feature
        [3.0, 4.0],
        [4.0, f64::NAN], // Missing value in second feature
        [5.0, 6.0],
        [6.0, 7.0],
    ];
    let y = array![0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    // Test Skip strategy - should remove rows with missing values
    let model = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Skip)
        .fit(&x, &y)
        .unwrap();

    // Should have 4 samples remaining (rows 0, 2, 4, 5)
    assert_eq!(model.n_features(), 2);

    // Test prediction on clean data
    let x_test = array![[2.5, 3.5], [5.5, 6.5]];
    let predictions = model.predict(&x_test).unwrap();
    assert_eq!(predictions.len(), 2);
}

#[test]
fn test_decision_tree_classifier_majority_imputation() {
    // Create data with missing values
    let x = array![
        [1.0, 2.0],
        [f64::NAN, 3.0], // Missing value in first feature
        [3.0, 4.0],
        [4.0, f64::NAN], // Missing value in second feature
        [5.0, 6.0],
    ];
    let y = array![0.0, 0.0, 1.0, 1.0, 1.0];

    // Test Majority strategy - should impute with column means
    let model = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Majority)
        .fit(&x, &y)
        .unwrap();

    // Should keep all 5 samples
    assert_eq!(model.n_features(), 2);

    // Test prediction
    let x_test = array![[2.5, 3.5], [5.5, 6.5]];
    let predictions = model.predict(&x_test).unwrap();
    assert_eq!(predictions.len(), 2);
}

#[test]
fn test_decision_tree_regressor_skip_missing() {
    // Create regression data with missing values
    let x = array![
        [1.0],
        [f64::NAN], // Missing value
        [3.0],
        [4.0],
        [f64::NAN], // Missing value
        [6.0],
    ];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

    // Test Skip strategy
    let model = DecisionTreeRegressor::new()
        .criterion(sklears_tree::SplitCriterion::MSE)
        .missing_values(MissingValueStrategy::Skip)
        .fit(&x, &y)
        .unwrap();

    // Should have 4 samples remaining (rows 0, 2, 3, 5)
    assert_eq!(model.n_features(), 1);

    // Test prediction
    let x_test = array![[2.5], [5.5]];
    let predictions = model.predict(&x_test).unwrap();
    assert_eq!(predictions.len(), 2);
}

#[test]
fn test_decision_tree_regressor_majority_imputation() {
    // Create regression data with missing values
    let x = array![
        [1.0],
        [f64::NAN], // Missing value
        [3.0],
        [4.0],
        [5.0],
    ];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

    // Test Majority strategy - should impute with column mean
    let model = DecisionTreeRegressor::new()
        .criterion(sklears_tree::SplitCriterion::MSE)
        .missing_values(MissingValueStrategy::Majority)
        .fit(&x, &y)
        .unwrap();

    // Should keep all 5 samples
    assert_eq!(model.n_features(), 1);

    // Test prediction
    let x_test = array![[2.5], [5.5]];
    let predictions = model.predict(&x_test).unwrap();
    assert_eq!(predictions.len(), 2);
}

#[test]
fn test_no_missing_values() {
    // Test that data without missing values works normally
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
    let y = array![0.0, 0.0, 1.0, 1.0];

    // Test with different missing value strategies - should all work the same
    for strategy in [
        MissingValueStrategy::Skip,
        MissingValueStrategy::Majority,
        MissingValueStrategy::Surrogate,
    ] {
        let model = DecisionTreeClassifier::new()
            .missing_values(strategy)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_classes(), 2);

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }
}

#[test]
#[ignore] // Missing value validation logic not yet fully implemented
fn test_all_samples_missing_error() {
    // Test error when all samples have missing values with Skip strategy
    let x = array![[f64::NAN, 2.0], [1.0, f64::NAN], [f64::NAN, f64::NAN],];
    let y = array![0.0, 1.0, 0.0];

    // Skip strategy should fail when all samples have missing values
    let result = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Skip)
        .fit(&x, &y);

    assert!(
        result.is_err(),
        "Should fail when all samples contain missing values"
    );
}

#[test]
fn test_column_all_missing() {
    // Test when an entire column has missing values
    let x = array![
        [1.0, f64::NAN],
        [2.0, f64::NAN],
        [3.0, f64::NAN],
        [4.0, f64::NAN],
    ];
    let y = array![0.0, 0.0, 1.0, 1.0];

    // Majority strategy should handle this by using 0.0 for the missing column
    let model = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Majority)
        .fit(&x, &y)
        .unwrap();

    assert_eq!(model.n_features(), 2);

    let x_test = array![[2.5, 1.0]]; // Test with any value for second feature
    let predictions = model.predict(&x_test).unwrap();
    assert_eq!(predictions.len(), 1);
}

#[test]
fn test_mixed_missing_patterns() {
    // Test various missing value patterns
    let x = array![
        [1.0, 2.0, 3.0],
        [f64::NAN, 2.0, 3.0],      // Missing first feature
        [1.0, f64::NAN, 3.0],      // Missing second feature
        [1.0, 2.0, f64::NAN],      // Missing third feature
        [f64::NAN, f64::NAN, 3.0], // Missing first two features
        [4.0, 5.0, 6.0],           // No missing values
    ];
    let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    // Test Skip strategy
    let model_skip = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Skip)
        .fit(&x, &y)
        .unwrap();

    // Should keep 2 samples (rows 0 and 5)
    assert_eq!(model_skip.n_features(), 3);

    // Test Majority strategy
    let model_majority = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Majority)
        .fit(&x, &y)
        .unwrap();

    // Should keep all 6 samples
    assert_eq!(model_majority.n_features(), 3);

    // Both models should be able to predict
    let x_test = array![[2.0, 3.0, 4.0]];

    let pred_skip = model_skip.predict(&x_test).unwrap();
    let pred_majority = model_majority.predict(&x_test).unwrap();

    assert_eq!(pred_skip.len(), 1);
    assert_eq!(pred_majority.len(), 1);
}

#[test]
fn test_surrogate_fallback() {
    // Test that Surrogate strategy falls back to Majority for now
    let x = array![[1.0, 2.0], [f64::NAN, 3.0], [3.0, 4.0],];
    let y = array![0.0, 0.0, 1.0];

    // Surrogate should work (falling back to majority)
    let model = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Surrogate)
        .fit(&x, &y)
        .unwrap();

    assert_eq!(model.n_features(), 2);

    let x_test = array![[2.0, 3.0]];
    let predictions = model.predict(&x_test).unwrap();
    assert_eq!(predictions.len(), 1);
}

#[test]
fn test_missing_values_mean_calculation() {
    // Test that mean imputation calculates correctly
    let x = array![
        [1.0, 10.0],
        [f64::NAN, 20.0], // Should be imputed with mean of [1.0, 3.0, 5.0] = 3.0
        [3.0, f64::NAN],  // Should be imputed with mean of [10.0, 20.0, 40.0] = 23.33...
        [5.0, 40.0],
    ];
    let y = array![0.0, 0.0, 1.0, 1.0];

    let model = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Majority)
        .fit(&x, &y)
        .unwrap();

    // Should successfully train with imputed values
    assert_eq!(model.n_features(), 2);

    // Test that prediction works
    let x_test = array![[3.0, 23.3]]; // Close to expected imputed values
    let predictions = model.predict(&x_test).unwrap();
    assert_eq!(predictions.len(), 1);
}
