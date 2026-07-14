//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;
#[allow(unused_imports)]
use std::collections::HashSet;

use super::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_2 {
    use super::*;
    use proptest::prelude::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Predict;
    #[test]
    fn test_bagging_classifier_creation() {
        let classifier = BaggingClassifier::new()
            .n_estimators(20)
            .random_state(42)
            .oob_score(true);
        assert_eq!(classifier.config.n_estimators, 20);
        assert_eq!(classifier.config.random_state, Some(42));
        assert!(classifier.config.oob_score);
    }
    #[test]
    fn test_bagging_classifier_fit_predict() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2, 0, 1];
        let classifier = BaggingClassifier::new().n_estimators(5).random_state(42);
        let fitted = classifier
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let predictions = fitted.predict(&x).expect("prediction should succeed");
        assert_eq!(predictions.len(), 8);
        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes().len(), 3);
        assert_eq!(fitted.n_features_in(), 2);
    }
    #[test]
    fn test_bagging_classifier_with_oob() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2, 0, 1, 2, 0];
        let classifier = BaggingClassifier::new()
            .n_estimators(10)
            .random_state(42)
            .oob_score(true)
            .bootstrap(true);
        let fitted = classifier
            .fit(&x, &y)
            .expect("model fitting should succeed");
        assert!(fitted.oob_score().is_some());
        let oob_score = fitted.oob_score().expect("operation should succeed");
        assert!((0.0..=1.0).contains(&oob_score));
        let predictions = fitted.predict(&x).expect("prediction should succeed");
        assert_eq!(predictions.len(), 10);
    }
    #[test]
    fn test_bagging_classifier_feature_bagging() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2];
        let classifier = BaggingClassifier::new()
            .n_estimators(5)
            .max_features(Some(2))
            .bootstrap_features(false)
            .random_state(42);
        let fitted = classifier
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let predictions = fitted.predict(&x).expect("prediction should succeed");
        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.n_features_in(), 4);
        let importances = fitted.feature_importances();
        assert_eq!(importances.len(), 4);
        let sum: Float = importances.sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_bagging_classifier_confidence_intervals() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![0, 0, 1, 1];
        let classifier = BaggingClassifier::new()
            .n_estimators(10)
            .random_state(42)
            .confidence_level(0.95);
        let fitted = classifier
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let (predictions, confidence_intervals) = fitted
            .predict_with_confidence(&x)
            .expect("operation should succeed");
        assert_eq!(predictions.len(), 4);
        assert_eq!(confidence_intervals.dim(), (4, 2));
        for i in 0..4 {
            assert!(confidence_intervals[[i, 0]] <= confidence_intervals[[i, 1]]);
        }
    }
    #[test]
    fn test_bagging_regressor_creation() {
        let regressor = BaggingRegressor::new().n_estimators(15).random_state(123);
        assert_eq!(regressor.config.n_estimators, 15);
        assert_eq!(regressor.config.random_state, Some(123));
    }
    fn regression_dataset() -> (Array2<Float>, Array1<Float>) {
        let x = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 2.0],
            [5.0, 3.0],
            [6.0, 3.0],
            [7.0, 4.0],
            [8.0, 4.0],
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
            [4.0, 8.0],
        ];
        let y = array![1.1, 2.9, 4.05, 5.95, 7.1, 8.9, 10.05, 11.95, -2.9, -2.1, -0.95, -0.05,];
        (x, y)
    }
    fn r2_score(y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Float {
        let mean_y = y_true.iter().sum::<Float>() / y_true.len() as Float;
        let ss_tot: Float = y_true.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: Float = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yi, &pi)| (yi - pi).powi(2))
            .sum();
        1.0 - ss_res / ss_tot
    }
    #[test]
    fn test_bagging_regressor_fit_predict_accuracy() {
        let (x, y) = regression_dataset();
        let fitted = BaggingRegressor::new()
            .n_estimators(25)
            .max_depth(Some(5))
            .random_state(42)
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let predictions = fitted.predict(&x).expect("prediction should succeed");
        assert_eq!(predictions.len(), x.nrows());
        let p_min = predictions
            .iter()
            .cloned()
            .fold(Float::INFINITY, Float::min);
        let p_max = predictions
            .iter()
            .cloned()
            .fold(Float::NEG_INFINITY, Float::max);
        assert!(
            p_max - p_min > 1.0,
            "predictions should vary, got range [{p_min}, {p_max}]"
        );
        let r2 = r2_score(&y, &predictions);
        assert!(r2 > 0.9, "training R^2 too low: {r2}");
    }
    #[test]
    fn test_bagging_regressor_oob_score() {
        let (x, y) = regression_dataset();
        let fitted = BaggingRegressor::new()
            .n_estimators(25)
            .max_depth(Some(5))
            .oob_score(true)
            .bootstrap(true)
            .random_state(42)
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let oob = fitted.oob_score();
        assert!(oob.is_some(), "oob_score should be computed when requested");
        let oob = oob.expect("oob score should be present");
        assert!(oob <= 1.0, "R^2 OOB score must not exceed 1.0, got {oob}");
    }
    #[test]
    fn test_bagging_regressor_different_seeds_differ() {
        let (x, y) = regression_dataset();
        let fitted_a = BaggingRegressor::new()
            .n_estimators(10)
            .max_depth(Some(4))
            .bootstrap(true)
            .random_state(1)
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let fitted_b = BaggingRegressor::new()
            .n_estimators(10)
            .max_depth(Some(4))
            .bootstrap(true)
            .random_state(999)
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let pred_a = fitted_a.predict(&x).expect("prediction should succeed");
        let pred_b = fitted_b.predict(&x).expect("prediction should succeed");
        let any_diff = pred_a
            .iter()
            .zip(pred_b.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-9);
        assert!(any_diff, "different seeds produced identical predictions");
    }
    #[test]
    fn test_bagging_regressor_determinism_same_seed() {
        let (x, y) = regression_dataset();
        let build = || {
            BaggingRegressor::new()
                .n_estimators(10)
                .max_depth(Some(4))
                .bootstrap(true)
                .random_state(7)
                .fit(&x, &y)
                .expect("model fitting should succeed")
        };
        let pred_1 = build().predict(&x).expect("prediction should succeed");
        let pred_2 = build().predict(&x).expect("prediction should succeed");
        assert_eq!(pred_1.len(), pred_2.len());
        for (a, b) in pred_1.iter().zip(pred_2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "same seed must produce bit-identical predictions"
            );
        }
    }
    #[test]
    fn test_bagging_regressor_bootstrap_diversity() {
        let x = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 2.0],
            [5.0, 3.0],
            [6.0, 3.0],
            [7.0, 4.0],
            [8.0, 4.0],
            [9.0, 5.0],
            [10.0, 5.0],
        ];
        let y = array![1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 10.0, 12.0, 13.0, 15.0];
        let fitted = BaggingRegressor::new()
            .n_estimators(8)
            .bootstrap(true)
            .random_state(42)
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let estimators_samples = fitted.estimators_samples();
        let mut unique_sample_sets = HashSet::new();
        for samples in estimators_samples {
            let mut sorted_samples = samples.clone();
            sorted_samples.sort();
            unique_sample_sets.insert(sorted_samples);
        }
        assert!(
            unique_sample_sets.len() > 1,
            "bootstrap samples lack diversity: {} unique of {} estimators",
            unique_sample_sets.len(),
            estimators_samples.len()
        );
    }
    #[test]
    fn test_bagging_regressor_invalid_input() {
        let regressor = BaggingRegressor::new();
        let x = Array2::zeros((0, 2));
        let y = Array1::zeros(0);
        assert!(regressor.fit(&x, &y).is_err());
        let regressor = BaggingRegressor::new();
        let x = Array2::zeros((3, 2));
        let y = Array1::zeros(2);
        assert!(regressor.fit(&x, &y).is_err());
    }
    #[test]
    fn test_bagging_regressor_feature_mismatch() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y_train = array![1.0, 2.0, 3.0];
        let x_test = array![[1.0, 2.0]];
        let fitted = BaggingRegressor::new()
            .fit(&x_train, &y_train)
            .expect("model fitting should succeed");
        assert!(fitted.predict(&x_test).is_err());
    }
    #[test]
    fn test_bagging_regressor_feature_importances_sum() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let fitted = BaggingRegressor::new()
            .n_estimators(5)
            .max_features(Some(2))
            .bootstrap_features(false)
            .random_state(42)
            .fit(&x, &y)
            .expect("model fitting should succeed");
        let importances = fitted.feature_importances();
        assert_eq!(importances.len(), 4);
        let sum: Float = importances.sum();
        assert!((sum - 1.0).abs() < 1e-10, "feature importances sum = {sum}");
    }
    #[test]
    fn test_bagging_config_default() {
        let config = BaggingConfig::default();
        assert_eq!(config.n_estimators, 10);
        assert!(config.bootstrap);
        assert!(!config.bootstrap_features);
        assert!(!config.oob_score);
        assert_eq!(config.random_state, None);
        assert_eq!(config.min_samples_split, 2);
        assert_eq!(config.min_samples_leaf, 1);
        assert_eq!(config.confidence_level, 0.95);
    }
    #[test]
    fn test_bagging_classifier_invalid_input() {
        let classifier = BaggingClassifier::new();
        let x = Array2::zeros((0, 2));
        let y = Array1::zeros(0);
        assert!(classifier.fit(&x, &y).is_err());
        let classifier = BaggingClassifier::new();
        let x = Array2::zeros((3, 2));
        let y = Array1::zeros(2);
        assert!(classifier.fit(&x, &y).is_err());
        let classifier = BaggingClassifier::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 0];
        assert!(classifier.fit(&x, &y).is_err());
    }
    #[test]
    fn test_bagging_classifier_feature_mismatch() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y_train = array![0, 1];
        let x_test = array![[1.0, 2.0]];
        let classifier = BaggingClassifier::new();
        let fitted = classifier
            .fit(&x_train, &y_train)
            .expect("model fitting should succeed");
        assert!(fitted.predict(&x_test).is_err());
    }
    proptest! {
        #[test] fn prop_bagging_deterministic_with_seed(n_estimators in 1usize..10,
        random_seed in 0u64..1000,) { let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
        [4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],]; let y = array![0,
        0, 1, 1, 2, 2, 0, 1]; let classifier1 = BaggingClassifier::new()
        .n_estimators(n_estimators).random_state(random_seed).fit(& x, & y)
        .expect("operation should succeed"); let classifier2 = BaggingClassifier::new()
        .n_estimators(n_estimators).random_state(random_seed).fit(& x, & y)
        .expect("operation should succeed"); let pred1 = classifier1.predict(& x)
        .expect("prediction should succeed"); let pred2 = classifier2.predict(& x)
        .expect("prediction should succeed"); prop_assert_eq!(pred1, pred2); } #[test] fn
        prop_bagging_feature_importance_normalization(n_estimators in 1usize..10,
        max_features in 1usize..4,) { let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0], [6.0, 7.0, 8.0],]; let y =
        array![0, 0, 1, 1, 2, 2]; let classifier = BaggingClassifier::new()
        .n_estimators(n_estimators).max_features(Some(max_features)).random_state(42)
        .fit(& x, & y).expect("operation should succeed"); let importances = classifier
        .feature_importances(); let sum : Float = importances.sum(); prop_assert!((sum -
        1.0).abs() < 1e-10); for & importance in importances.iter() {
        prop_assert!(importance >= 0.0); } } #[test] fn
        prop_bagging_bootstrap_diversity(n_estimators in 2usize..8,) { let x =
        array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0],
        [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0],]; let y = array![0, 0, 1, 1,
        2, 2, 0, 1, 2, 0]; let classifier = BaggingClassifier::new()
        .n_estimators(n_estimators).bootstrap(true).random_state(42).fit(& x, & y)
        .expect("operation should succeed"); let estimators_samples = classifier
        .estimators_samples(); let mut unique_sample_sets = HashSet::new(); for samples
        in estimators_samples { let mut sorted_samples = samples.clone(); sorted_samples
        .sort(); unique_sample_sets.insert(sorted_samples); } prop_assert!(!
        unique_sample_sets.is_empty()); } #[test] fn
        prop_bagging_prediction_stability(n_estimators in 3usize..10,) { let x =
        array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0],];
        let y = array![0, 0, 1, 1, 2, 2]; let classifier = BaggingClassifier::new()
        .n_estimators(n_estimators).random_state(42).fit(& x, & y)
        .expect("operation should succeed"); let predictions = classifier.predict(& x)
        .expect("prediction should succeed"); let classes = classifier.classes(); for &
        pred in predictions.iter() { prop_assert!(classes.iter().any(|& c | c == pred));
        } prop_assert_eq!(predictions.len(), x.nrows()); } #[test] fn
        prop_bagging_oob_score_bounds(n_estimators in 5usize..15,) { let x = array![[1.0,
        2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0],
        [8.0, 9.0], [9.0, 10.0], [10.0, 11.0], [11.0, 12.0], [12.0, 13.0],]; let y =
        array![0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2]; let classifier =
        BaggingClassifier::new().n_estimators(n_estimators).oob_score(true)
        .bootstrap(true).random_state(42).fit(& x, & y)
        .expect("operation should succeed"); if let Some(oob_score) = classifier
        .oob_score() { prop_assert!((0.0..= 1.0).contains(& oob_score)); } } #[test] fn
        prop_bagging_confidence_intervals_bounds(n_estimators in 3usize..8,
        confidence_level in 0.7..0.99,) { let x = array![[1.0, 2.0], [2.0, 3.0], [3.0,
        4.0], [4.0, 5.0],]; let y = array![0, 0, 1, 1]; let classifier =
        BaggingClassifier::new().n_estimators(n_estimators)
        .confidence_level(confidence_level).random_state(42).fit(& x, & y)
        .expect("operation should succeed"); let (predictions, confidence_intervals) =
        classifier.predict_with_confidence(& x).expect("operation should succeed"); for i
        in 0..predictions.len() { let lower = confidence_intervals[[i, 0]]; let upper =
        confidence_intervals[[i, 1]]; prop_assert!(lower <= upper); prop_assert!(lower
        .is_finite() && upper.is_finite()); } }
    }
}
