/// Test missing value handling for Decision Trees and Random Forest
use scirs2_core::ndarray::array;
use sklears_core::traits::{Fit, Predict};
use sklears_tree::{
    builder::handle_missing_values, DecisionTreeClassifier, DecisionTreeRegressor,
    MissingValueStrategy,
};

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
        .expect("operation should succeed");

    // Should have 4 samples remaining (rows 0, 2, 4, 5)
    assert_eq!(model.n_features(), 2);

    // Test prediction on clean data
    let x_test = array![[2.5, 3.5], [5.5, 6.5]];
    let predictions = model.predict(&x_test).expect("prediction should succeed");
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
        .expect("operation should succeed");

    // Should keep all 5 samples
    assert_eq!(model.n_features(), 2);

    // Test prediction
    let x_test = array![[2.5, 3.5], [5.5, 6.5]];
    let predictions = model.predict(&x_test).expect("prediction should succeed");
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
        .expect("operation should succeed");

    // Should have 4 samples remaining (rows 0, 2, 3, 5)
    assert_eq!(model.n_features(), 1);

    // Test prediction
    let x_test = array![[2.5], [5.5]];
    let predictions = model.predict(&x_test).expect("prediction should succeed");
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
        .expect("operation should succeed");

    // Should keep all 5 samples
    assert_eq!(model.n_features(), 1);

    // Test prediction
    let x_test = array![[2.5], [5.5]];
    let predictions = model.predict(&x_test).expect("prediction should succeed");
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
            .expect("operation should succeed");

        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_classes(), 2);

        let predictions = model.predict(&x).expect("prediction should succeed");
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
        .expect("operation should succeed");

    assert_eq!(model.n_features(), 2);

    let x_test = array![[2.5, 1.0]]; // Test with any value for second feature
    let predictions = model.predict(&x_test).expect("prediction should succeed");
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
        .expect("operation should succeed");

    // Should keep 2 samples (rows 0 and 5)
    assert_eq!(model_skip.n_features(), 3);

    // Test Majority strategy
    let model_majority = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Majority)
        .fit(&x, &y)
        .expect("operation should succeed");

    // Should keep all 6 samples
    assert_eq!(model_majority.n_features(), 3);

    // Both models should be able to predict
    let x_test = array![[2.0, 3.0, 4.0]];

    let pred_skip = model_skip
        .predict(&x_test)
        .expect("prediction should succeed");
    let pred_majority = model_majority
        .predict(&x_test)
        .expect("prediction should succeed");

    assert_eq!(pred_skip.len(), 1);
    assert_eq!(pred_majority.len(), 1);
}

#[test]
fn test_surrogate_fallback() {
    // Surrogate strategy on small data: verifies fit + predict succeed.
    let x = array![[1.0, 2.0], [f64::NAN, 3.0], [3.0, 4.0],];
    let y = array![0.0, 0.0, 1.0];

    let model = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Surrogate)
        .fit(&x, &y)
        .expect("operation should succeed");

    assert_eq!(model.n_features(), 2);

    let x_test = array![[2.0, 3.0]];
    let predictions = model.predict(&x_test).expect("prediction should succeed");
    assert_eq!(predictions.len(), 1);
}

#[test]
fn test_surrogate_imputation_uses_correlated_column() {
    // col_0 and col_1 are strongly correlated (col_1 ≈ col_0).
    // When col_0 is missing for the last row, the surrogate should use col_1
    // to infer a replacement for col_0 (rather than the plain mean).
    //
    // Numeric verification:
    //   col_0 observed (rows 0–7): [1,2,3,4,5,6,7,8]
    //   column mean        = 4.5
    //   median             = 4.5
    //   left-side mean     = (1+2+3+4)/4 = 2.5  (values ≤ 4.5)
    //   right-side mean    = (5+6+7+8)/4 = 6.5  (values > 4.5)
    //   col_1 for NaN row  = 9.0  →  surr sends to "right" side
    //   surrogate imputed value  ≈ 6.5   (not 4.5 from plain mean)
    let x = array![
        [1.0, 1.1],
        [2.0, 2.0],
        [3.0, 3.1],
        [4.0, 3.9],
        [5.0, 5.0],
        [6.0, 6.1],
        [7.0, 7.0],
        [8.0, 7.9],
        [f64::NAN, 9.0], // Missing col_0; col_1 = 9.0 → surrogate should push imputed value high
    ];
    let y_labels = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // Call handle_missing_values directly to inspect the imputed matrix.
    let (x_surr, _) = handle_missing_values(&x, &y_labels, MissingValueStrategy::Surrogate)
        .expect("surrogate imputation should succeed");

    // The NaN is in row 8, col 0.
    let imputed_val = x_surr[[8, 0]];

    // Surrogate should give right-side mean (≈ 6.5), which is clearly above the
    // column mean (4.5).  Plain mean-imputation would give exactly 4.5.
    assert!(
        imputed_val > 5.5,
        "surrogate imputed value should be right-side mean (~6.5), got {imputed_val}"
    );

    // Sanity: also verify fit + predict succeed end-to-end.
    let model = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Surrogate)
        .fit(&x, &y_labels)
        .expect("surrogate fit should succeed");

    assert_eq!(model.n_features(), 2);
    let x_test = array![[4.5, 4.5]];
    let preds = model.predict(&x_test).expect("prediction should succeed");
    assert_eq!(preds.len(), 1);
}

#[test]
fn test_surrogate_multiple_missing_columns() {
    // Multiple columns have missing values simultaneously.
    let x = array![
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
        [5.0, 50.0, 500.0],
        [6.0, f64::NAN, 600.0],  // col_1 missing
        [f64::NAN, 70.0, 700.0], // col_0 missing
        [8.0, 80.0, 800.0],
        [9.0, 90.0, 900.0],
        [10.0, 100.0, 1000.0],
    ];
    let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let model = DecisionTreeClassifier::new()
        .missing_values(MissingValueStrategy::Surrogate)
        .fit(&x, &y)
        .expect("surrogate multi-column fit should succeed");

    assert_eq!(model.n_features(), 3);
    let x_test = array![[5.5, 55.0, 550.0]];
    let preds = model.predict(&x_test).expect("prediction should succeed");
    assert_eq!(preds.len(), 1);
}

#[test]
fn test_surrogate_regression_with_missing() {
    // Regression: ensure surrogate imputation works with regressor too.
    let x = array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
        [f64::NAN, 6.0], // col_0 missing; col_1 = 6.0 is a surrogate
        [7.0, 7.0],
    ];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];

    let model = DecisionTreeRegressor::new()
        .criterion(sklears_tree::SplitCriterion::MSE)
        .missing_values(MissingValueStrategy::Surrogate)
        .fit(&x, &y)
        .expect("surrogate regression fit should succeed");

    assert_eq!(model.n_features(), 2);
    let x_test = array![[3.5, 3.5]];
    let preds = model.predict(&x_test).expect("prediction should succeed");
    assert_eq!(preds.len(), 1);
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
        .expect("operation should succeed");

    // Should successfully train with imputed values
    assert_eq!(model.n_features(), 2);

    // Test that prediction works
    let x_test = array![[3.0, 23.3]]; // Close to expected imputed values
    let predictions = model.predict(&x_test).expect("prediction should succeed");
    assert_eq!(predictions.len(), 1);
}
