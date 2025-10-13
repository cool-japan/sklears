//! Comprehensive tests for voting ensemble methods

use super::*;
use crate::voting::{
    config::VotingStrategy,
    core::VotingClassifier,
    ensemble::{ensemble_utils, MockEstimator},
    simd_ops::*,
    strategies::*,
};
use scirs2_core::ndarray::array;
use sklears_core::prelude::Float;
use sklears_core::traits::{Fit, Predict, Untrained};

#[test]
fn test_voting_classifier_creation() {
    let classifier = VotingClassifier::<Untrained>::new(VotingClassifierConfig::default());
    assert_eq!(classifier.estimators().len(), 0);
}

#[test]
fn test_voting_classifier_builder() {
    let classifier = VotingClassifier::builder()
        .voting(VotingStrategy::Soft)
        .weights(vec![1.0, 2.0, 1.5])
        .confidence_weighting(true)
        .confidence_threshold(0.8)
        .build();

    assert_eq!(classifier.config().voting, VotingStrategy::Soft);
    assert_eq!(classifier.config().weights, Some(vec![1.0, 2.0, 1.5]));
    assert!(classifier.config().confidence_weighting);
    assert_eq!(classifier.config().confidence_threshold, 0.8);
}

#[test]
fn test_convenience_builders() {
    use crate::voting::core::VotingClassifierBuilder;

    let confidence_classifier = VotingClassifierBuilder::confidence_weighted().build();
    assert_eq!(
        confidence_classifier.config().voting,
        VotingStrategy::ConfidenceWeighted
    );

    let bayesian_classifier = VotingClassifierBuilder::bayesian_averaging().build();
    assert_eq!(
        bayesian_classifier.config().voting,
        VotingStrategy::BayesianAveraging
    );

    let entropy_classifier = VotingClassifierBuilder::entropy_weighted().build();
    assert_eq!(
        entropy_classifier.config().voting,
        VotingStrategy::EntropyWeighted
    );
}

#[test]
fn test_mock_estimator() {
    let mut estimator = MockEstimator::new(0.5);
    assert_eq!(estimator.weight(), 1.0);
    assert_eq!(estimator.performance(), 0.8);
    assert!(estimator.supports_proba());
    assert!(estimator.is_fitted());

    estimator.set_weight(0.7);
    estimator.update_performance(0.9);
    assert_eq!(estimator.weight(), 0.7);
    assert_eq!(estimator.performance(), 0.9);
}

#[test]
fn test_mock_estimator_predictions() {
    let estimator = MockEstimator::new(0.0);
    let x = array![[1.0, 2.0], [0.0, -1.0], [-1.0, 1.0]];

    let predictions = estimator.predict(&x).unwrap();
    assert_eq!(predictions.len(), 3);

    let probabilities = estimator.predict_proba(&x).unwrap();
    assert_eq!(probabilities.dim(), (3, 2));

    // Check probabilities sum to 1
    for i in 0..3 {
        let row_sum = probabilities.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_mock_estimator_uncertainty() {
    let estimator = MockEstimator::new(0.2);
    let x = array![[1.0, 2.0], [0.0, 0.0], [-1.0, -1.0]];

    let uncertainty = estimator.uncertainty(&x).unwrap();
    assert_eq!(uncertainty.len(), 3);

    // Uncertainty should be non-negative
    for &unc in uncertainty.iter() {
        assert!(unc >= 0.0);
    }
}

#[test]
fn test_voting_classifier_fit_predict() {
    let mut classifier = VotingClassifier::builder()
        .voting(VotingStrategy::Hard)
        .build();

    // Add some mock estimators
    classifier.add_estimator(Box::new(MockEstimator::new(0.1)));
    classifier.add_estimator(Box::new(MockEstimator::new(-0.1)));
    classifier.add_estimator(Box::new(MockEstimator::new(0.0)));

    let x = array![[1.0, 2.0], [2.0, 3.0], [0.0, 1.0], [-1.0, -2.0]];
    let y = array![1.0, 1.0, 0.0, 0.0];

    let fitted_classifier = classifier.fit(&x, &y).unwrap();

    assert_eq!(fitted_classifier.n_features_in(), 2);
    assert_eq!(fitted_classifier.classes().len(), 2);

    let predictions = fitted_classifier.predict(&x).unwrap();
    assert_eq!(predictions.len(), 4);
}

#[test]
fn test_voting_strategies() {
    let mut classifier = VotingClassifier::builder()
        .voting(VotingStrategy::Soft)
        .build();

    classifier.add_estimator(Box::new(MockEstimator::new(0.2)));
    classifier.add_estimator(Box::new(MockEstimator::new(-0.2)));

    let x = array![[1.0, 2.0], [0.0, 0.0]];
    let y = array![1.0, 0.0];

    let fitted_classifier = classifier.fit(&x, &y).unwrap();

    // Test that prediction doesn't panic for different strategies
    let test_strategies = vec![
        VotingStrategy::Hard,
        VotingStrategy::Soft,
        VotingStrategy::Weighted,
        VotingStrategy::ConfidenceWeighted,
    ];

    for strategy in test_strategies {
        let mut test_classifier = VotingClassifier::builder().voting(strategy).build();

        test_classifier.add_estimator(Box::new(MockEstimator::new(0.1)));
        test_classifier.add_estimator(Box::new(MockEstimator::new(-0.1)));

        let fitted = test_classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x);
        assert!(predictions.is_ok());
    }
}

#[test]
fn test_predict_with_confidence() {
    let mut classifier = VotingClassifier::builder()
        .voting(VotingStrategy::Hard)
        .build();

    classifier.add_estimator(Box::new(MockEstimator::new(0.1)));
    classifier.add_estimator(Box::new(MockEstimator::new(-0.1)));
    classifier.add_estimator(Box::new(MockEstimator::new(0.0)));

    let x = array![[1.0, 2.0], [0.0, 0.0], [-1.0, -1.0]];
    let y = array![1.0, 0.0, 0.0];

    let fitted_classifier = classifier.fit(&x, &y).unwrap();

    let (predictions, confidence) = fitted_classifier.predict_with_confidence(&x).unwrap();

    assert_eq!(predictions.len(), 3);
    assert_eq!(confidence.len(), 3);

    // Confidence should be in [0, 1]
    for &conf in confidence.iter() {
        assert!(conf >= 0.0 && conf <= 1.0);
    }
}

#[test]
fn test_predict_proba() {
    let mut classifier = VotingClassifier::builder()
        .voting(VotingStrategy::Soft)
        .build();

    classifier.add_estimator(Box::new(MockEstimator::new(0.1)));
    classifier.add_estimator(Box::new(MockEstimator::new(-0.1)));

    let x = array![[1.0, 2.0], [0.0, 0.0]];
    let y = array![1.0, 0.0];

    let fitted_classifier = classifier.fit(&x, &y).unwrap();

    let probabilities = fitted_classifier.predict_proba(&x).unwrap();

    assert_eq!(probabilities.dim(), (2, 2));

    // Probabilities should sum to 1 for each sample
    for i in 0..2 {
        let row_sum = probabilities.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_update_weights_dynamically() {
    let mut classifier = VotingClassifier::builder()
        .weight_adjustment_rate(0.2)
        .build();

    classifier.add_estimator(Box::new(MockEstimator::new(0.1)));
    classifier.add_estimator(Box::new(MockEstimator::new(-0.1)));

    let x = array![[1.0, 2.0], [0.0, 0.0]];
    let y = array![1.0, 0.0];

    let mut fitted_classifier = classifier.fit(&x, &y).unwrap();

    let initial_weights: Vec<Float> = fitted_classifier
        .estimators()
        .iter()
        .map(|e| e.weight())
        .collect();

    // Update with different performances
    let performances = vec![0.9, 0.6];
    fitted_classifier
        .update_weights_dynamically(&performances)
        .unwrap();

    let updated_weights: Vec<Float> = fitted_classifier
        .estimators()
        .iter()
        .map(|e| e.weight())
        .collect();

    // Weights should have changed
    assert_ne!(initial_weights, updated_weights);

    // Weights should sum to approximately 1
    let weight_sum: Float = updated_weights.iter().sum();
    assert!((weight_sum - 1.0).abs() < 1e-6);
}

// SIMD Operation Tests
#[test]
fn test_simd_mean_f32() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = simd_mean_f32(&data);
    assert!((mean - 3.0).abs() < 1e-6);

    let empty_data = vec![];
    let empty_mean = simd_mean_f32(&empty_data);
    assert_eq!(empty_mean, 0.0);
}

#[test]
fn test_simd_sum_f32() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sum = simd_sum_f32(&data);
    assert!((sum - 15.0).abs() < 1e-6);
}

#[test]
fn test_simd_variance_f32() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = simd_mean_f32(&data);
    let variance = simd_variance_f32(&data, mean);

    // Expected variance for [1,2,3,4,5] with mean 3 is 2.5
    assert!((variance - 2.5).abs() < 1e-6);

    let single_data = vec![5.0];
    let single_variance = simd_variance_f32(&single_data, 5.0);
    assert_eq!(single_variance, 0.0);
}

#[test]
fn test_simd_entropy_f32() {
    let uniform_probs = vec![0.25, 0.25, 0.25, 0.25];
    let entropy = simd_entropy_f32(&uniform_probs);

    // Entropy of uniform distribution over 4 outcomes is log(4) ≈ 1.386
    // Formula: -sum(p * log(p)) = -(4 * 0.25 * ln(0.25)) = -4 * 0.25 * (-1.386) = 1.386
    assert!((entropy - (-(4.0 * 0.25 * (0.25_f32).ln()))).abs() < 1e-6);

    let certain_probs = vec![1.0, 0.0, 0.0, 0.0];
    let certain_entropy = simd_entropy_f32(&certain_probs);
    assert!(certain_entropy.abs() < 1e-6);
}

#[test]
fn test_simd_weighted_sum_f32() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let weights = vec![0.1, 0.2, 0.3, 0.4];
    let mut output = vec![0.0; 4];

    simd_weighted_sum_f32(&values, &weights, &mut output);

    for i in 0..4 {
        assert!((output[i] - values[i] * weights[i]).abs() < 1e-6);
    }
}

#[test]
fn test_simd_argmax_f32() {
    let values = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let max_idx = simd_argmax_f32(&values);
    assert_eq!(max_idx, 1); // Index of value 5.0

    let empty_values = vec![];
    let empty_max_idx = simd_argmax_f32(&empty_values);
    assert_eq!(empty_max_idx, 0);
}

#[test]
fn test_simd_normalize_f32() {
    let input = vec![3.0, 4.0, 0.0];
    let mut output = vec![0.0; 3];

    simd_normalize_f32(&input, &mut output);

    // L2 norm of [3,4,0] is 5, so normalized should be [0.6, 0.8, 0.0]
    assert!((output[0] - 0.6).abs() < 1e-6);
    assert!((output[1] - 0.8).abs() < 1e-6);
    assert!((output[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_simd_hard_voting_weighted() {
    let pred1 = array![0.0, 1.0, 0.0];
    let pred2 = array![1.0, 1.0, 0.0];
    let pred3 = array![0.0, 0.0, 1.0];

    let all_predictions = vec![pred1, pred2, pred3];
    let weights = vec![1.0, 2.0, 1.0];
    let classes = vec![0.0, 1.0];

    let result = simd_hard_voting_weighted(&all_predictions, &weights, &classes);

    assert_eq!(result.len(), 3);
    // First sample: 0 gets weight 2 (1+1), 1 gets weight 2 -> tie, returns first (0)
    // Second sample: all vote for 1 -> 1 wins
    // Third sample: 0 gets weight 3, 1 gets weight 1 -> 0 wins
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 1.0);
    assert_eq!(result[2], 0.0);
}

#[test]
fn test_simd_soft_voting_weighted() {
    let prob1 = array![[0.3, 0.7], [0.8, 0.2]];
    let prob2 = array![[0.4, 0.6], [0.6, 0.4]];

    let all_probabilities = vec![prob1, prob2];
    let weights = vec![1.0, 2.0];

    let result = simd_soft_voting_weighted(&all_probabilities, &weights);

    assert_eq!(result.dim(), (2, 2));

    // Check probabilities sum to 1 for each sample
    for i in 0..2 {
        let row_sum = result.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

// Strategy Tests
#[test]
fn test_weighted_average_f32() {
    let values = vec![1.0, 2.0, 3.0];
    let weights = vec![0.5, 0.3, 0.2];

    let result = weighted_average_f32(&values, &weights);
    let expected = (1.0 * 0.5 + 2.0 * 0.3 + 3.0 * 0.2) / (0.5 + 0.3 + 0.2);
    assert!((result - expected).abs() < 1e-6);
}

#[test]
fn test_consensus_voting() {
    let pred1 = array![0.0, 1.0, 0.0, 1.0];
    let pred2 = array![0.0, 1.0, 1.0, 1.0];
    let pred3 = array![0.0, 0.0, 0.0, 1.0];

    let all_predictions = vec![pred1, pred2, pred3];
    let result = consensus_voting(&all_predictions, 0.6).unwrap();

    assert_eq!(result.len(), 4);
    assert_eq!(result[0], 0.0); // Unanimous for 0
    assert_eq!(result[3], 1.0); // Unanimous for 1
}

#[test]
fn test_dynamic_weight_adjustment() {
    let current_weights = vec![0.3, 0.3, 0.4];
    let performances = vec![0.9, 0.6, 0.8];
    let learning_rate = 0.1;

    let new_weights =
        dynamic_weight_adjustment(&current_weights, &performances, learning_rate).unwrap();

    assert_eq!(new_weights.len(), 3);

    // Weights should sum to 1
    let weight_sum: Float = new_weights.iter().sum();
    assert!((weight_sum - 1.0).abs() < 1e-6);

    // Better performing estimator (0.9) should get higher weight
    assert!(new_weights[0] > new_weights[1]);
}

#[test]
fn test_temperature_scaled_voting() {
    let prob1 = array![[0.3, 0.7], [0.8, 0.2]];
    let prob2 = array![[0.4, 0.6], [0.6, 0.4]];

    let all_probabilities = vec![prob1, prob2];
    let temperature = 0.5; // Lower temperature makes distribution more peaked

    let result = temperature_scaled_voting(&all_probabilities, temperature).unwrap();

    assert_eq!(result.dim(), (2, 2));

    // Check probabilities sum to 1 for each sample
    for i in 0..2 {
        let row_sum = result.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_rank_based_voting() {
    let prob1 = array![[0.1, 0.9], [0.8, 0.2]];
    let prob2 = array![[0.2, 0.8], [0.7, 0.3]];
    let prob3 = array![[0.3, 0.7], [0.6, 0.4]];

    let all_probabilities = vec![prob1, prob2, prob3];

    let result = rank_based_voting(&all_probabilities).unwrap();

    assert_eq!(result.len(), 2);

    // Results should be valid class indices (0 or 1 for binary classification)
    for &prediction in result.iter() {
        assert!(prediction == 0.0 || prediction == 1.0);
    }
}

// Ensemble utility tests
#[test]
fn test_ensemble_diversity_calculation() {
    let estimator1 = Box::new(MockEstimator::new(0.1)) as Box<dyn EnsembleMember + Send + Sync>;
    let estimator2 = Box::new(MockEstimator::new(-0.1)) as Box<dyn EnsembleMember + Send + Sync>;
    let estimator3 = Box::new(MockEstimator::new(0.0)) as Box<dyn EnsembleMember + Send + Sync>;

    let estimators = vec![estimator1, estimator2, estimator3];
    let x = array![[1.0, 2.0], [0.0, 0.0], [-1.0, -1.0]];

    let diversity = ensemble_utils::calculate_ensemble_diversity(&estimators, &x).unwrap();

    assert!(diversity >= 0.0 && diversity <= 1.0);
}

#[test]
fn test_ensemble_weight_update() {
    let mut estimator1 = Box::new(MockEstimator::new(0.1)) as Box<dyn EnsembleMember + Send + Sync>;
    let mut estimator2 =
        Box::new(MockEstimator::new(-0.1)) as Box<dyn EnsembleMember + Send + Sync>;

    estimator1.set_weight(0.5);
    estimator2.set_weight(0.5);

    let mut estimators = vec![estimator1, estimator2];
    let performances = vec![0.8, 0.9];
    let learning_rate = 0.2;

    ensemble_utils::update_ensemble_weights(&mut estimators, &performances, learning_rate);

    let weights: Vec<Float> = estimators.iter().map(|e| e.weight()).collect();

    // Higher performing estimator should get higher weight
    assert!(weights[1] > weights[0]);

    // Weights should sum to approximately 1
    let weight_sum: Float = weights.iter().sum();
    assert!((weight_sum - 1.0).abs() < 0.1);
}

#[test]
fn test_ensemble_stats() {
    let estimator1 = Box::new(
        MockEstimator::new(0.1)
            .with_weight(0.3)
            .with_performance(0.8),
    ) as Box<dyn EnsembleMember + Send + Sync>;
    let estimator2 = Box::new(
        MockEstimator::new(-0.1)
            .with_weight(0.7)
            .with_performance(0.9),
    ) as Box<dyn EnsembleMember + Send + Sync>;

    let estimators = vec![estimator1, estimator2];

    let stats = ensemble_utils::get_ensemble_stats(&estimators);

    assert_eq!(stats.n_estimators, 2);
    assert!((stats.mean_weight - 0.5).abs() < 1e-6);
    assert!((stats.mean_performance - 0.85).abs() < 1e-6);
    assert!(stats.weight_variance > 0.0);
}

// Property-based test-like behavior
#[test]
fn test_voting_classifier_properties() {
    // Test various configurations don't panic
    let strategies = vec![
        VotingStrategy::Hard,
        VotingStrategy::Soft,
        VotingStrategy::Weighted,
        VotingStrategy::ConfidenceWeighted,
        VotingStrategy::BayesianAveraging,
    ];

    for strategy in strategies {
        let mut classifier = VotingClassifier::builder().voting(strategy).build();

        // Add estimators with different biases
        classifier.add_estimator(Box::new(MockEstimator::new(0.2)));
        classifier.add_estimator(Box::new(MockEstimator::new(-0.2)));
        classifier.add_estimator(Box::new(MockEstimator::new(0.0)));

        let x = array![[1.0, 2.0], [-1.0, -2.0], [0.0, 1.0]];
        let y = array![1.0, 0.0, 1.0];

        let fitted_classifier = classifier.fit(&x, &y).unwrap();
        let predictions = fitted_classifier.predict(&x).unwrap();

        assert_eq!(predictions.len(), 3);

        // Predictions should be valid class labels
        for &pred in predictions.iter() {
            assert!(pred == 0.0 || pred == 1.0);
        }
    }
}

#[test]
fn test_error_handling() {
    let classifier = VotingClassifier::builder().build();
    let x = array![[1.0, 2.0]];
    let y = array![1.0, 0.0]; // Mismatched dimensions

    // Should fail with shape mismatch
    assert!(classifier.fit(&x, &y).is_err());

    // Test with empty estimators
    let empty_classifier = VotingClassifier::builder().build();
    let x = array![[1.0, 2.0]];
    let y = array![1.0];

    let fitted = empty_classifier.fit(&x, &y).unwrap();
    // Prediction should fail with no estimators
    assert!(fitted.predict(&x).is_err());
}

#[test]
fn test_feature_mismatch() {
    let mut classifier = VotingClassifier::builder().build();
    classifier.add_estimator(Box::new(MockEstimator::new(0.0)));

    let x_train = array![[1.0, 2.0], [0.0, 1.0]];
    let y_train = array![1.0, 0.0];

    let fitted = classifier.fit(&x_train, &y_train).unwrap();

    // Try to predict with wrong number of features
    let x_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2

    assert!(fitted.predict(&x_wrong).is_err());
}

#[test]
fn test_large_ensemble() {
    let mut classifier = VotingClassifier::builder()
        .voting(VotingStrategy::Hard)
        .build();

    // Add many estimators
    for i in 0..20 {
        let bias = (i as f64 - 10.0) / 10.0; // Range from -1.0 to 1.0
        classifier.add_estimator(Box::new(MockEstimator::new(bias)));
    }

    let x = array![[1.0, 2.0], [0.0, 0.0], [-1.0, -1.0], [2.0, 3.0]];
    let y = array![1.0, 0.0, 0.0, 1.0];

    let fitted = classifier.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.estimators().len(), 20);
}
