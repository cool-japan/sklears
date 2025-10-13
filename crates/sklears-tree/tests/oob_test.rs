/// Test out-of-bag estimation for Random Forest
use scirs2_core::ndarray::array;
use sklears_core::traits::{Fit, Predict};
use sklears_tree::RandomForestClassifier;

#[test]
fn test_random_forest_oob_score() {
    let x = array![
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
    ];
    let y = array![0, 0, 1, 1, 2, 2];

    // Test Random Forest with OOB score enabled
    let model = RandomForestClassifier::new()
        .n_estimators(10)
        .oob_score(true)
        .bootstrap(true)
        .random_state(42)
        .fit(&x, &y)
        .unwrap();

    // Check that we can access basic properties
    assert_eq!(model.n_features(), 2);
    assert_eq!(model.n_classes(), 3);

    // Check that OOB score was computed
    let oob_score = model.oob_score();
    assert!(
        oob_score.is_some(),
        "OOB score should be computed when oob_score=true"
    );

    let score_value = oob_score.unwrap();
    assert!(
        (0.0..=1.0).contains(&score_value),
        "OOB score should be between 0 and 1, got {}",
        score_value
    );

    // Check that OOB decision function was computed
    let oob_decision = model.oob_decision_function();
    assert!(
        oob_decision.is_some(),
        "OOB decision function should be computed when oob_score=true"
    );

    let decision_values = oob_decision.unwrap();
    assert_eq!(decision_values.shape(), &[6, 3]); // 6 samples, 3 classes

    // Test predictions work
    let predictions = model.predict(&x).unwrap();
    assert_eq!(predictions.len(), 6);
}

#[test]
fn test_random_forest_without_oob_score() {
    let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];
    let y = array![0, 0, 1, 1];

    // Test Random Forest with OOB score disabled (default)
    let model = RandomForestClassifier::new()
        .n_estimators(5)
        .fit(&x, &y)
        .unwrap();

    // Check that OOB score was not computed
    assert!(
        model.oob_score().is_none(),
        "OOB score should not be computed when oob_score=false"
    );
    assert!(
        model.oob_decision_function().is_none(),
        "OOB decision function should not be computed when oob_score=false"
    );

    // Test predictions still work
    let predictions = model.predict(&x).unwrap();
    assert_eq!(predictions.len(), 4);
}

#[test]
fn test_random_forest_oob_without_bootstrap() {
    let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];
    let y = array![0, 0, 1, 1];

    // Test Random Forest with OOB score enabled but bootstrap disabled
    // This should not compute OOB score since bootstrap is required
    let model = RandomForestClassifier::new()
        .n_estimators(5)
        .oob_score(true)
        .bootstrap(false)
        .fit(&x, &y)
        .unwrap();

    // Check that OOB score was not computed
    assert!(
        model.oob_score().is_none(),
        "OOB score should not be computed when bootstrap=false"
    );
    assert!(
        model.oob_decision_function().is_none(),
        "OOB decision function should not be computed when bootstrap=false"
    );
}
