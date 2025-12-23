use super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Array1, Array2};
use sklears_core::prelude::{Fit, Predict};
use sklears_core::types::Float;

#[test]
fn test_adaboost_decision_function() {
    let x = array![[1.0, 2.0], [2.0, 1.0],];
    let y = array![0.0, 1.0];
    let ada = AdaBoostClassifier::new()
        .n_estimators(5)
        .fit(&x, &y)
        .unwrap();
    let decision = ada.decision_function(&x).unwrap();
    assert_eq!(decision.dim(), (2, 1));
}
#[test]
fn test_adaboost_empty_data() {
    let x = Array2::<Float>::zeros((0, 2));
    let y = Array1::<Float>::zeros(0);
    let result = AdaBoostClassifier::new().fit(&x, &y);
    assert!(result.is_err());
}
#[test]
fn test_adaboost_single_class() {
    let x = array![[1.0, 2.0], [2.0, 3.0],];
    let y = array![0.0, 0.0];
    let result = AdaBoostClassifier::new().fit(&x, &y);
    assert!(result.is_err());
}
#[test]
fn test_adaboost_config_builder() {
    let ada = AdaBoostClassifier::new()
        .n_estimators(100)
        .learning_rate(0.5)
        .random_state(123)
        .algorithm(AdaBoostAlgorithm::SAMMER);
    assert_eq!(ada.config.n_estimators, 100);
    assert_eq!(ada.config.learning_rate, 0.5);
    assert_eq!(ada.config.random_state, Some(123));
    assert!(matches!(ada.config.algorithm, AdaBoostAlgorithm::SAMMER));
}
#[test]
fn test_adaboost_feature_mismatch() {
    let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],];
    let y_train = array![0.0, 1.0];
    let x_test = array![[1.0, 2.0],];
    let ada = AdaBoostClassifier::new().fit(&x_train, &y_train).unwrap();
    let result = ada.predict(&x_test);
    assert!(result.is_err());
}
#[test]
fn test_adaboost_samme_r_algorithm() {
    let x = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 2.0],
        [1.0, 0.5],
        [2.0, 1.5],
        [5.0, 3.0],
        [6.0, 4.0],
    ];
    let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let ada_r = AdaBoostClassifier::new()
        .n_estimators(5)
        .learning_rate(0.8)
        .with_samme_r()
        .random_state(42)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(ada_r.classes().len(), 2);
    assert_eq!(ada_r.n_classes(), 2);
    assert_eq!(ada_r.n_features_in(), 2);
    let predictions = ada_r.predict(&x).unwrap();
    assert_eq!(predictions.len(), 8);
    let probabilities = ada_r.predict_proba(&x).unwrap();
    assert_eq!(probabilities.dim(), (8, 2));
    for i in 0..8 {
        let prob_sum = probabilities.row(i).sum();
        assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
    }
    let decision_scores = ada_r.decision_function(&x).unwrap();
    assert_eq!(decision_scores.dim(), (8, 1));
    let ada_samme = AdaBoostClassifier::new()
        .n_estimators(5)
        .learning_rate(0.8)
        .algorithm(AdaBoostAlgorithm::SAMME)
        .random_state(42)
        .fit(&x, &y)
        .unwrap();
    let predictions_samme = ada_samme.predict(&x).unwrap();
    assert_eq!(predictions_samme.len(), 8);
    let importances = ada_r.feature_importances().unwrap();
    assert_eq!(importances.len(), 2);
    assert_abs_diff_eq!(importances.sum(), 1.0, epsilon = 1e-10);
}
#[test]
fn test_adaboost_samme_r_multiclass() {
    let x = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 2.0],
        [1.0, 0.5],
        [2.0, 1.5],
        [5.0, 3.0],
        [6.0, 4.0],
        [7.0, 1.0],
    ];
    let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    let ada_r = AdaBoostClassifier::new()
        .n_estimators(4)
        .learning_rate(0.7)
        .with_samme_r()
        .random_state(123)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(ada_r.classes().len(), 3);
    assert_eq!(ada_r.n_classes(), 3);
    let predictions = ada_r.predict(&x).unwrap();
    assert_eq!(predictions.len(), 9);
    let probabilities = ada_r.predict_proba(&x).unwrap();
    assert_eq!(probabilities.dim(), (9, 3));
    for i in 0..9 {
        let prob_sum = probabilities.row(i).sum();
        assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
    }
    let decision_scores = ada_r.decision_function(&x).unwrap();
    assert_eq!(decision_scores.dim(), (9, 3));
}
#[test]
fn test_adaboost_real_algorithm() {
    let x = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 2.0],
        [1.0, 0.5],
        [2.0, 1.5],
    ];
    let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let ada_real = AdaBoostClassifier::new()
        .n_estimators(5)
        .learning_rate(1.0)
        .algorithm(AdaBoostAlgorithm::RealAdaBoost)
        .random_state(42)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(ada_real.classes().len(), 2);
    assert_eq!(ada_real.n_classes(), 2);
    assert_eq!(ada_real.n_features_in(), 2);
    let predictions = ada_real.predict(&x).unwrap();
    assert_eq!(predictions.len(), 6);
    for &pred in predictions.iter() {
        assert!(ada_real.classes().iter().any(|&c| c == pred));
    }
    let probabilities = ada_real.predict_proba(&x).unwrap();
    assert_eq!(probabilities.dim(), (6, 2));
    for i in 0..6 {
        let prob_sum = probabilities.row(i).sum();
        assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
    }
    let decision_scores = ada_real.decision_function(&x).unwrap();
    assert_eq!(decision_scores.len(), 6);
    let weights = ada_real.estimator_weights();
    assert!(weights.len() > 0);
    assert!(weights.iter().all(|&w| w.is_finite()));
}
#[test]
fn test_real_adaboost_multiclass_error() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![0.0, 1.0, 2.0];
    let result = AdaBoostClassifier::new()
        .algorithm(AdaBoostAlgorithm::RealAdaBoost)
        .fit(&x, &y);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("binary classification"));
}
