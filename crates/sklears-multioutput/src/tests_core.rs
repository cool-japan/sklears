//! Core tests for multi-output classifiers, regressors, chains, multi-label, and metrics

use super::*;
use crate::utilities::CLARE;
use scirs2_core::ndarray::{array, Array2};
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::Float;

#[test]
fn test_multi_output_classifier() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[0, 1], [1, 0], [1, 1], [0, 0]];

    let moc = MultiOutputClassifier::new();
    let fitted = moc
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    assert_eq!(fitted.n_targets(), 2);
    assert_eq!(fitted.classes().len(), 2);

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // Check that predictions are valid (within the classes for each target)
    for target_idx in 0..2 {
        let target_classes = &fitted.classes()[target_idx];
        for sample_idx in 0..4 {
            let pred = predictions[[sample_idx, target_idx]];
            assert!(target_classes.contains(&pred));
        }
    }
}

#[test]
fn test_multi_output_regressor() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[1.5, 2.5], [2.5, 3.5], [2.0, 1.5], [1.0, 1.5]];

    let mor = MultiOutputRegressor::new();
    let fitted = mor
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    assert_eq!(fitted.n_targets(), 2);

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // Predictions should be finite numbers
    for pred in predictions.iter() {
        assert!(pred.is_finite());
    }
}

#[test]
fn test_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[0, 1], [1, 0], [0, 1]]; // Wrong number of rows

    let moc = MultiOutputClassifier::new();
    assert!(moc.fit(&X.view(), &y).is_err());
}

#[test]
fn test_empty_targets() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = Array2::<i32>::zeros((2, 0)); // No targets

    let moc = MultiOutputClassifier::new();
    assert!(moc.fit(&X.view(), &y).is_err());
}

#[test]
fn test_prediction_shape_mismatch() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[0, 1], [1, 0]];

    let moc = MultiOutputClassifier::new();
    let fitted = moc
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Test with wrong number of features
    let X_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
    assert!(fitted.predict(&X_wrong.view()).is_err());
}

#[test]
fn test_classifier_chain() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[0, 1], [1, 0], [1, 1], [0, 0]];

    let cc = ClassifierChain::new();
    let fitted = cc
        .fit_simple(&X.view(), &y)
        .expect("operation should succeed");

    assert_eq!(fitted.n_targets(), 2);
    assert_eq!(fitted.chain_order(), &[0, 1]); // Default order

    let predictions = fitted
        .predict_simple(&X.view())
        .expect("operation should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // Check that predictions are valid
    for sample_idx in 0..4 {
        for target_idx in 0..2 {
            let pred = predictions[[sample_idx, target_idx]];
            // Predictions should be either 0 or 1 for binary classification
            assert!(pred == 0 || pred == 1);
        }
    }
}

#[test]
fn test_classifier_chain_custom_order() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[0, 1], [1, 0], [1, 1]];

    let cc = ClassifierChain::new().order(vec![1, 0]); // Reverse order
    let fitted = cc
        .fit_simple(&X.view(), &y)
        .expect("operation should succeed");

    assert_eq!(fitted.chain_order(), &[1, 0]);

    let predictions = fitted
        .predict_simple(&X.view())
        .expect("operation should succeed");
    assert_eq!(predictions.dim(), (3, 2));
}

#[test]
fn test_classifier_chain_invalid_order() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[0, 1], [1, 0]];

    let cc = ClassifierChain::new().order(vec![0, 1, 2]); // Too many indices
    assert!(cc.fit_simple(&X.view(), &y).is_err());
}

#[test]
fn test_classifier_chain_monte_carlo() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[0, 1], [1, 0], [1, 1], [0, 0]];

    let cc = ClassifierChain::new();
    let fitted = cc
        .fit_simple(&X.view(), &y)
        .expect("operation should succeed");

    // Test Monte Carlo predictions with probabilities
    let mc_probs = fitted
        .predict_monte_carlo(&X.view(), 100, Some(42))
        .expect("operation should succeed");
    assert_eq!(mc_probs.dim(), (4, 2));

    // All probabilities should be between 0 and 1
    for prob in mc_probs.iter() {
        assert!(*prob >= 0.0 && *prob <= 1.0);
    }

    // Test Monte Carlo predictions with labels
    let mc_labels = fitted
        .predict_monte_carlo_labels(&X.view(), 100, Some(42))
        .expect("operation should succeed");
    assert_eq!(mc_labels.dim(), (4, 2));

    // All predictions should be binary (0 or 1)
    for pred in mc_labels.iter() {
        assert!(*pred == 0 || *pred == 1);
    }

    // Test reproducibility with same random state
    let mc_probs2 = fitted
        .predict_monte_carlo(&X.view(), 100, Some(42))
        .expect("operation should succeed");
    for (i, (&prob1, &prob2)) in mc_probs.iter().zip(mc_probs2.iter()).enumerate() {
        assert!(
            (prob1 - prob2).abs() < 1e-10,
            "Probabilities should be identical with same random state at index {}",
            i
        );
    }
}

#[test]
fn test_classifier_chain_monte_carlo_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[0, 1], [1, 0]];

    let cc = ClassifierChain::new();
    let fitted = cc
        .fit_simple(&X.view(), &y)
        .expect("operation should succeed");

    // Test with zero samples
    assert!(fitted.predict_monte_carlo(&X.view(), 0, None).is_err());

    // Test with wrong number of features
    let X_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
    assert!(fitted
        .predict_monte_carlo(&X_wrong.view(), 10, None)
        .is_err());
}

#[test]
fn test_regressor_chain() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[1.5, 2.5], [2.5, 3.5], [2.0, 1.5], [1.0, 1.5]];

    let rc = RegressorChain::new();
    let fitted = rc
        .fit_simple(&X.view(), &y)
        .expect("operation should succeed");

    assert_eq!(fitted.n_targets(), 2);
    assert_eq!(fitted.chain_order(), &[0, 1]); // Default order

    let predictions = fitted
        .predict_simple(&X.view())
        .expect("operation should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // Predictions should be finite numbers
    for pred in predictions.iter() {
        assert!(pred.is_finite());
    }
}

#[test]
fn test_regressor_chain_custom_order() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[1.5, 2.5], [2.5, 3.5], [2.0, 1.5]];

    let rc = RegressorChain::new().order(vec![1, 0]); // Reverse order
    let fitted = rc
        .fit_simple(&X.view(), &y)
        .expect("operation should succeed");

    assert_eq!(fitted.chain_order(), &[1, 0]);

    let predictions = fitted
        .predict_simple(&X.view())
        .expect("operation should succeed");
    assert_eq!(predictions.dim(), (3, 2));

    // Predictions should be finite numbers
    for pred in predictions.iter() {
        assert!(pred.is_finite());
    }
}

#[test]
fn test_regressor_chain_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1.5, 2.5], [2.5, 3.5], [2.0, 1.5]]; // Wrong number of rows

    let rc = RegressorChain::new();
    assert!(rc.fit_simple(&X.view(), &y).is_err());
}

#[test]
fn test_regressor_chain_invalid_order() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1.5, 2.5], [2.5, 3.5]];

    let rc = RegressorChain::new().order(vec![0, 1, 2]); // Too many indices
    assert!(rc.fit_simple(&X.view(), &y).is_err());
}

#[test]
fn test_binary_relevance() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label binary

    let br = BinaryRelevance::new();
    let fitted = br.fit(&X.view(), &y).expect("model fitting should succeed");

    assert_eq!(fitted.n_labels(), 2);
    assert_eq!(fitted.classes().len(), 2);

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // Check that predictions are binary (0 or 1)
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }

    // Test probability predictions
    let probabilities = fitted
        .predict_proba(&X.view())
        .expect("operation should succeed");
    assert_eq!(probabilities.dim(), (4, 2));

    // Check that probabilities are in [0, 1]
    for prob in probabilities.iter() {
        assert!(*prob >= 0.0 && *prob <= 1.0);
    }
}

#[test]
fn test_binary_relevance_single_label() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[1], [0], [1]]; // Single binary label

    let br = BinaryRelevance::new();
    let fitted = br.fit(&X.view(), &y).expect("model fitting should succeed");

    assert_eq!(fitted.n_labels(), 1);

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (3, 1));

    // Check that predictions are binary
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }
}

#[test]
fn test_binary_relevance_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Wrong number of rows

    let br = BinaryRelevance::new();
    assert!(br.fit(&X.view(), &y).is_err());
}

#[test]
fn test_binary_relevance_non_binary_labels() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[0, 1], [1, 2], [2, 0]]; // Non-binary labels

    let br = BinaryRelevance::new();
    assert!(br.fit(&X.view(), &y).is_err());
}

#[test]
fn test_binary_relevance_predict_shape_mismatch() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1]];

    let br = BinaryRelevance::new();
    let fitted = br.fit(&X.view(), &y).expect("model fitting should succeed");

    // Test with wrong number of features
    let X_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
    assert!(fitted.predict(&X_wrong.view()).is_err());
    assert!(fitted.predict_proba(&X_wrong.view()).is_err());
}

#[test]
fn test_label_powerset() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label binary combinations

    let lp = LabelPowerset::new();
    let fitted = lp.fit(&X.view(), &y).expect("model fitting should succeed");

    assert_eq!(fitted.n_labels(), 2);
    assert_eq!(fitted.n_classes(), 4); // 4 unique combinations: [1,0], [0,1], [1,1], [0,0]

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // Check that predictions are binary (0 or 1)
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }

    // Test decision function
    let scores = fitted
        .decision_function(&X.view())
        .expect("operation should succeed");
    assert_eq!(scores.dim(), (4, 4)); // 4 samples, 4 classes

    // Scores should be finite
    for score in scores.iter() {
        assert!(score.is_finite());
    }
}

#[test]
fn test_label_powerset_simple_case() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[1, 0], [0, 1], [1, 0]]; // Only 2 unique combinations

    let lp = LabelPowerset::new();
    let fitted = lp.fit(&X.view(), &y).expect("model fitting should succeed");

    assert_eq!(fitted.n_labels(), 2);
    assert_eq!(fitted.n_classes(), 2); // Only 2 unique combinations: [1,0], [0,1]

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (3, 2));

    // Check that predictions are binary
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }
}

#[test]
fn test_label_powerset_single_label() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[1], [0], [1]]; // Single binary label

    let lp = LabelPowerset::new();
    let fitted = lp.fit(&X.view(), &y).expect("model fitting should succeed");

    assert_eq!(fitted.n_labels(), 1);
    assert_eq!(fitted.n_classes(), 2); // 2 unique combinations: [1], [0]

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (3, 1));

    // Check that predictions are binary
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }
}

#[test]
fn test_label_powerset_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Wrong number of rows

    let lp = LabelPowerset::new();
    assert!(lp.fit(&X.view(), &y).is_err());
}

#[test]
fn test_label_powerset_non_binary_labels() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[0, 1], [1, 2], [2, 0]]; // Non-binary labels

    let lp = LabelPowerset::new();
    assert!(lp.fit(&X.view(), &y).is_err());
}

#[test]
fn test_label_powerset_predict_shape_mismatch() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1]];

    let lp = LabelPowerset::new();
    let fitted = lp.fit(&X.view(), &y).expect("model fitting should succeed");

    // Test with wrong number of features
    let X_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
    assert!(fitted.predict(&X_wrong.view()).is_err());
    assert!(fitted.decision_function(&X_wrong.view()).is_err());
}

#[test]
fn test_label_powerset_all_same_combination() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[1, 0], [1, 0], [1, 0]]; // All samples have same label combination

    let lp = LabelPowerset::new();
    let fitted = lp.fit(&X.view(), &y).expect("model fitting should succeed");

    assert_eq!(fitted.n_labels(), 2);
    assert_eq!(fitted.n_classes(), 1); // Only 1 unique combination

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (3, 2));

    // All predictions should be [1, 0]
    for sample_idx in 0..3 {
        assert_eq!(predictions[[sample_idx, 0]], 1);
        assert_eq!(predictions[[sample_idx, 1]], 0);
    }
}

#[test]
fn test_pruned_label_powerset_default_strategy() {
    // Test data with some rare combinations
    let X = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [1.5, 2.5],
        [2.5, 1.5]
    ];
    let y = array![
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0], // Frequent combinations
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1], // More frequent ones
    ];

    let plp = PrunedLabelPowerset::new()
        .min_frequency(2)
        .strategy(PruningStrategy::DefaultMapping(vec![0, 0]));

    let fitted = plp
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Should have pruned to only frequent combinations
    assert!(fitted.n_frequent_classes() <= 4); // At most [1,0], [0,1], [1,1], [0,0]
    assert_eq!(fitted.min_frequency(), 2);

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (8, 2));

    // All predictions should be binary
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }
}

#[test]
fn test_pruned_label_powerset_similarity_strategy() {
    // Test with similarity mapping strategy
    let X = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ];
    let y = array![
        [1, 0],
        [1, 0],
        [1, 0], // Frequent: [1, 0] appears 3 times
        [0, 1],
        [0, 1], // Frequent: [0, 1] appears 2 times
        [1, 1]  // Rare: [1, 1] appears 1 time
    ];

    let plp = PrunedLabelPowerset::new()
        .min_frequency(2)
        .strategy(PruningStrategy::SimilarityMapping);

    let fitted = plp
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Should have only 2 frequent combinations: [1,0] and [0,1]
    assert_eq!(fitted.n_frequent_classes(), 2);

    // The rare combination [1,1] should be mapped to one of the frequent ones
    let mapping = fitted.combination_mapping();
    let rare_combo = vec![1, 1];
    assert!(mapping.contains_key(&rare_combo));

    // The mapped combination should be one of the frequent ones
    let mapped = mapping.get(&rare_combo).expect("index should be valid");
    assert!(mapped == &vec![1, 0] || mapped == &vec![0, 1]);

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (6, 2));

    // All predictions should be binary
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }
}

#[test]
fn test_pruned_label_powerset_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[0, 1], [1, 0]];

    // Test with minimum frequency that results in no frequent combinations
    let plp = PrunedLabelPowerset::new().min_frequency(5); // Too high
    assert!(plp.fit(&X.view(), &y).is_err());

    // Test with invalid default combination length
    let plp = PrunedLabelPowerset::new().strategy(PruningStrategy::DefaultMapping(vec![0, 1, 0])); // 3 elements for 2 labels
    assert!(plp.fit(&X.view(), &y).is_err());

    // Test with non-binary labels
    let y_bad = array![[2, 1], [1, 0]]; // Contains non-binary value
    let plp = PrunedLabelPowerset::new();
    assert!(plp.fit(&X.view(), &y_bad).is_err());
}

#[test]
fn test_pruned_label_powerset_edge_cases() {
    // Test with minimal data that meets frequency requirement
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [1, 0]]; // Same combination twice

    let plp = PrunedLabelPowerset::new().min_frequency(2);
    let fitted = plp
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Should have at least 1 combination, possibly 2 if default is added
    assert!(fitted.n_frequent_classes() >= 1);
    assert!(fitted.frequent_combinations().len() >= 1);

    // The frequent combinations should include [1, 0]
    assert!(fitted.frequent_combinations().contains(&vec![1, 0]));

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (2, 2));

    // All predictions should be [1, 0]
    for sample_idx in 0..2 {
        assert_eq!(predictions[[sample_idx, 0]], 1);
        assert_eq!(predictions[[sample_idx, 1]], 0);
    }
}

#[test]
fn test_metrics_hamming_loss() {
    let y_true = array![[1, 0, 1], [0, 1, 0], [1, 1, 1]];
    let y_pred = array![[1, 0, 0], [0, 1, 1], [1, 0, 1]]; // 3 errors out of 9

    let loss =
        metrics::hamming_loss(&y_true.view(), &y_pred.view()).expect("operation should succeed");
    assert!((loss - 3.0 / 9.0).abs() < 1e-10);
}

#[test]
fn test_metrics_subset_accuracy() {
    let y_true = array![[1, 0, 1], [0, 1, 0], [1, 1, 1]];
    let y_pred = array![[1, 0, 1], [0, 1, 1], [1, 0, 1]]; // Only first subset matches

    let accuracy =
        metrics::subset_accuracy(&y_true.view(), &y_pred.view()).expect("operation should succeed");
    assert!((accuracy - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_metrics_jaccard_score() {
    let y_true = array![[1, 0, 1], [0, 1, 0]];
    let y_pred = array![[1, 0, 0], [0, 1, 1]];

    let score =
        metrics::jaccard_score(&y_true.view(), &y_pred.view()).expect("operation should succeed");
    // Sample 1: intersection=1, union=2, jaccard=0.5
    // Sample 2: intersection=1, union=2, jaccard=0.5
    // Average: 0.5
    assert!((score - 0.5).abs() < 1e-10);
}

#[test]
fn test_metrics_f1_score_micro() {
    let y_true = array![[1, 0, 1], [0, 1, 0], [1, 1, 1]];
    let y_pred = array![[1, 0, 0], [0, 1, 1], [1, 0, 1]];

    let f1 = metrics::f1_score(&y_true.view(), &y_pred.view(), "micro")
        .expect("operation should succeed");
    // TP=4, FP=1, FN=2
    // Precision = 4/5 = 0.8, Recall = 4/6 = 0.6667
    // F1 = 2 * 0.8 * 0.6667 / (0.8 + 0.6667) = 0.727
    assert!((f1 - 0.7272727272727273).abs() < 1e-10);
}

#[test]
fn test_metrics_f1_score_macro() {
    let y_true = array![[1, 0], [0, 1], [1, 1]];
    let y_pred = array![[1, 0], [0, 1], [1, 0]]; // Perfect for label 0, imperfect for label 1

    let f1 = metrics::f1_score(&y_true.view(), &y_pred.view(), "macro")
        .expect("operation should succeed");
    // Label 0: TP=2, FP=0, FN=0 -> F1=1.0
    // Label 1: TP=1, FP=0, FN=1 -> Precision=1.0, Recall=0.5, F1=0.667
    // Macro average: (1.0 + 0.667) / 2 = 0.833
    assert!((f1 - 0.8333333333333334).abs() < 1e-10);
}

#[test]
fn test_metrics_f1_score_samples() {
    let y_true = array![[1, 0], [0, 1], [1, 1]];
    let y_pred = array![[1, 0], [0, 1], [1, 0]];

    let f1 = metrics::f1_score(&y_true.view(), &y_pred.view(), "samples")
        .expect("sampling should succeed");
    // Sample 0: TP=1, FP=0, FN=0 -> F1=1.0
    // Sample 1: TP=1, FP=0, FN=0 -> F1=1.0
    // Sample 2: TP=1, FP=0, FN=1 -> Precision=1.0, Recall=0.5, F1=0.667
    // Average: (1.0 + 1.0 + 0.667) / 3 = 0.889
    assert!((f1 - 0.8888888888888888).abs() < 1e-10);
}

#[test]
fn test_metrics_coverage_error() {
    let y_true = array![[1, 0, 1], [0, 1, 0]];
    let y_scores = array![[0.9, 0.1, 0.8], [0.2, 0.9, 0.3]];

    let coverage = metrics::coverage_error(&y_true.view(), &y_scores.view())
        .expect("operation should succeed");
    // Sample 0: sorted scores [0.9, 0.8, 0.1] -> labels [0, 2, 1]
    //          true labels are at positions 1 and 2, so coverage = 2
    // Sample 1: sorted scores [0.9, 0.3, 0.2] -> labels [1, 2, 0]
    //          true label is at position 1, so coverage = 1
    // Average: (2 + 1) / 2 = 1.5
    assert!((coverage - 1.5).abs() < 1e-10);
}

#[test]
fn test_metrics_label_ranking_average_precision() {
    let y_true = array![[1, 0, 1], [0, 1, 0]];
    let y_scores = array![[0.9, 0.1, 0.8], [0.2, 0.9, 0.3]];

    let lrap = metrics::label_ranking_average_precision(&y_true.view(), &y_scores.view())
        .expect("operation should succeed");
    // Sample 0: sorted scores [0.9, 0.8, 0.1] -> labels [0, 2, 1]
    //          true labels: 0 (pos 1), 2 (pos 2)
    //          precision at pos 1: 1/1=1.0, precision at pos 2: 2/2=1.0
    //          LRAP = (1.0 + 1.0) / 2 = 1.0
    // Sample 1: sorted scores [0.9, 0.3, 0.2] -> labels [1, 2, 0]
    //          true label: 1 (pos 1)
    //          precision at pos 1: 1/1=1.0
    //          LRAP = 1.0 / 1 = 1.0
    // Average: (1.0 + 1.0) / 2 = 1.0
    assert!((lrap - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_invalid_shapes() {
    let y_true = array![[1, 0], [0, 1]];
    let y_pred = array![[1, 0, 1]]; // Wrong shape

    assert!(metrics::hamming_loss(&y_true.view(), &y_pred.view()).is_err());
    assert!(metrics::subset_accuracy(&y_true.view(), &y_pred.view()).is_err());
    assert!(metrics::jaccard_score(&y_true.view(), &y_pred.view()).is_err());
    assert!(metrics::f1_score(&y_true.view(), &y_pred.view(), "micro").is_err());
}

#[test]
fn test_metrics_invalid_f1_average() {
    let y_true = array![[1, 0], [0, 1]];
    let y_pred = array![[1, 0], [0, 1]];

    assert!(metrics::f1_score(&y_true.view(), &y_pred.view(), "invalid").is_err());
}

#[test]
fn test_ensemble_of_chains() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[0, 1], [1, 0], [1, 1], [0, 0]];

    let eoc = EnsembleOfChains::new().n_chains(3).random_state(42);
    let fitted = eoc
        .fit_simple(&X.view(), &y)
        .expect("operation should succeed");

    assert_eq!(fitted.n_chains(), 3);
    assert_eq!(fitted.n_targets(), 2);

    let predictions = fitted
        .predict_simple(&X.view())
        .expect("operation should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // Check that predictions are binary (0 or 1)
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }

    // Test probability predictions
    let probabilities = fitted
        .predict_proba_simple(&X.view())
        .expect("operation should succeed");
    assert_eq!(probabilities.dim(), (4, 2));

    // Check that probabilities are in [0, 1]
    for prob in probabilities.iter() {
        assert!(*prob >= 0.0 && *prob <= 1.0);
    }
}

#[test]
fn test_ensemble_of_chains_single_chain() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1]];

    let eoc = EnsembleOfChains::new().n_chains(1);
    let fitted = eoc
        .fit_simple(&X.view(), &y)
        .expect("operation should succeed");

    assert_eq!(fitted.n_chains(), 1);

    let predictions = fitted
        .predict_simple(&X.view())
        .expect("operation should succeed");
    assert_eq!(predictions.dim(), (2, 2));
}

#[test]
fn test_ensemble_of_chains_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Wrong number of rows

    let eoc = EnsembleOfChains::new();
    assert!(eoc.fit_simple(&X.view(), &y).is_err());
}

#[test]
fn test_one_vs_rest_classifier() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label binary

    let ovr = OneVsRestClassifier::new();
    let fitted = ovr
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    assert_eq!(fitted.n_labels(), 2);
    assert_eq!(fitted.classes().len(), 2);

    let predictions = fitted
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // Check that predictions are binary (0 or 1)
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }

    // Test probability predictions
    let probabilities = fitted
        .predict_proba(&X.view())
        .expect("operation should succeed");
    assert_eq!(probabilities.dim(), (4, 2));

    // Check that probabilities are in [0, 1]
    for prob in probabilities.iter() {
        assert!(*prob >= 0.0 && *prob <= 1.0);
    }

    // Test decision function
    let scores = fitted
        .decision_function(&X.view())
        .expect("operation should succeed");
    assert_eq!(scores.dim(), (4, 2));

    // Scores should be finite
    for score in scores.iter() {
        assert!(score.is_finite());
    }
}

#[test]
fn test_one_vs_rest_classifier_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Wrong number of rows

    let ovr = OneVsRestClassifier::new();
    assert!(ovr.fit(&X.view(), &y).is_err());
}

#[test]
fn test_metrics_one_error() {
    let y_true = array![[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let y_scores = array![[0.9, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]];

    let one_err =
        metrics::one_error(&y_true.view(), &y_scores.view()).expect("operation should succeed");
    // All top-ranked labels are correct, so one-error should be 0
    assert!((one_err - 0.0).abs() < 1e-10);
}

#[test]
fn test_metrics_one_error_with_errors() {
    let y_true = array![[1, 0], [0, 1]];
    let y_scores = array![[0.3, 0.7], [0.6, 0.4]]; // Top predictions are wrong

    let one_err =
        metrics::one_error(&y_true.view(), &y_scores.view()).expect("operation should succeed");
    // Both samples have incorrect top predictions, so one-error should be 1.0
    assert!((one_err - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_ranking_loss() {
    let y_true = array![[1, 0], [0, 1]];
    let y_scores = array![[0.8, 0.2], [0.3, 0.7]]; // Correct ordering

    let ranking_loss =
        metrics::ranking_loss(&y_true.view(), &y_scores.view()).expect("operation should succeed");
    // Perfect ranking, so loss should be 0
    assert!((ranking_loss - 0.0).abs() < 1e-10);
}

#[test]
fn test_metrics_ranking_loss_with_errors() {
    let y_true = array![[1, 0], [0, 1]];
    let y_scores = array![[0.2, 0.8], [0.7, 0.3]]; // Incorrect ordering

    let ranking_loss =
        metrics::ranking_loss(&y_true.view(), &y_scores.view()).expect("operation should succeed");
    // All pairs are incorrectly ordered, so loss should be 1.0
    assert!((ranking_loss - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_average_precision_score() {
    let y_true = array![[1, 0, 1], [0, 1, 0]];
    let y_scores = array![[0.9, 0.1, 0.8], [0.2, 0.9, 0.3]];

    let ap_score = metrics::average_precision_score(&y_true.view(), &y_scores.view())
        .expect("operation should succeed");
    // With perfect ranking for both samples, AP should be 1.0
    assert!((ap_score - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_precision_recall_micro() {
    let y_true = array![[1, 0, 1], [0, 1, 0], [1, 1, 1]];
    let y_pred = array![[1, 0, 0], [0, 1, 1], [1, 0, 1]];

    let precision = metrics::precision_score_micro(&y_true.view(), &y_pred.view())
        .expect("operation should succeed");
    let recall = metrics::recall_score_micro(&y_true.view(), &y_pred.view())
        .expect("operation should succeed");

    // TP=4, FP=1, FN=2
    // Precision = 4/5 = 0.8, Recall = 4/6 = 0.6667
    assert!((precision - 0.8).abs() < 1e-10);
    assert!((recall - 0.6666666666666666).abs() < 1e-10);
}

#[test]
fn test_metrics_invalid_shapes_new_metrics() {
    let y_true = array![[1, 0], [0, 1]];
    let y_pred = array![[1, 0, 1]]; // Wrong shape
    let y_scores = array![[0.8, 0.2, 0.1]]; // Wrong shape

    assert!(metrics::one_error(&y_true.view(), &y_scores.view()).is_err());
    assert!(metrics::ranking_loss(&y_true.view(), &y_scores.view()).is_err());
    assert!(metrics::average_precision_score(&y_true.view(), &y_scores.view()).is_err());
    assert!(metrics::precision_score_micro(&y_true.view(), &y_pred.view()).is_err());
    assert!(metrics::recall_score_micro(&y_true.view(), &y_pred.view()).is_err());
}
