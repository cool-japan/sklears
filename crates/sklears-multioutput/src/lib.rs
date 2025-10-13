#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_doc_comments)]
#![allow(unused_parens)]
#![allow(unused_comparisons)]
//! Multi-output regression and classification
//!
//! This module provides meta-estimators for multi-target prediction problems.
//! It includes strategies for independent multi-output prediction.

// #![warn(missing_docs)]

pub mod activation;
pub mod adversarial;
pub mod chains;
pub mod classification;
pub mod core;
pub mod correlation;
pub mod ensemble;
pub mod hierarchical;
pub mod label_analysis;
pub mod loss;
pub mod metrics;
pub mod mlp;
pub mod multi_label;
pub mod multitask;
pub mod neighbors;
pub mod neural;
pub mod optimization;
pub mod probabilistic;
pub mod ranking;
pub mod recurrent;
pub mod regularization;
pub mod sequence;
pub mod sparse_storage;
pub mod svm;
pub mod transfer_learning;
pub mod tree;
pub mod utilities;
pub mod utils;

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)

// Re-export core multi-output algorithms
pub use core::{
    MultiOutputClassifier, MultiOutputClassifierTrained, MultiOutputRegressor,
    MultiOutputRegressorTrained,
};

// Re-export chain-based algorithms
pub use chains::{
    BayesianClassifierChain, BayesianClassifierChainTrained, ChainMethod, ClassifierChain,
    ClassifierChainTrained, EnsembleOfChains, EnsembleOfChainsTrained, RegressorChain,
    RegressorChainTrained,
};

// Re-export ensemble algorithms
pub use ensemble::{GradientBoostingMultiOutput, GradientBoostingMultiOutputTrained, WeakLearner};

// Re-export neural network algorithms
pub use neural::{
    ActivationFunction, AdversarialMultiTaskNetwork, AdversarialMultiTaskNetworkTrained,
    AdversarialStrategy, CellType, GradientReversalConfig, LambdaSchedule, LossFunction,
    MultiOutputMLP, MultiOutputMLPClassifier, MultiOutputMLPRegressor, MultiOutputMLPTrained,
    MultiTaskNeuralNetwork, MultiTaskNeuralNetworkTrained, RecurrentNeuralNetwork,
    RecurrentNeuralNetworkTrained, SequenceMode, TaskBalancing, TaskDiscriminator,
};

// Re-export adversarial learning types that are not in neural
pub use adversarial::AdversarialConfig;

// Re-export regularization algorithms
pub use regularization::{
    GroupLasso, GroupLassoTrained, MetaLearningMultiTask, MetaLearningMultiTaskTrained,
    MultiTaskElasticNet, MultiTaskElasticNetTrained, NuclearNormRegression,
    NuclearNormRegressionTrained, RegularizationStrategy, TaskClusteringRegressionTrained,
    TaskClusteringRegularization, TaskRelationshipLearning, TaskRelationshipLearningTrained,
    TaskSimilarityMethod,
};

// Re-export correlation and dependency analysis
pub use correlation::{
    CITestMethod, CITestResult, CITestResults, ConditionalIndependenceTester, CorrelationAnalysis,
    CorrelationType, DependencyGraph, DependencyGraphBuilder, DependencyMethod, GraphStatistics,
    OutputCorrelationAnalyzer,
};

// Re-export transfer learning algorithms
pub use transfer_learning::{
    ContinualLearning, ContinualLearningTrained, CrossTaskTransferLearning,
    CrossTaskTransferLearningTrained, DomainAdaptation, DomainAdaptationTrained,
    KnowledgeDistillation, KnowledgeDistillationTrained, ProgressiveTransferLearning,
    ProgressiveTransferLearningTrained,
};

// Re-export optimization algorithms
pub use optimization::{
    JointLossConfig, JointLossOptimizer, JointLossOptimizerTrained, LossCombination,
    LossFunction as OptimizationLossFunction, MultiObjectiveConfig, MultiObjectiveOptimizer,
    MultiObjectiveOptimizerTrained, NSGA2Algorithm, NSGA2Config, NSGA2Optimizer,
    NSGA2OptimizerTrained, ParetoSolution, ScalarizationConfig, ScalarizationMethod,
    ScalarizationOptimizer, ScalarizationOptimizerTrained,
};

// Re-export probabilistic algorithms
pub use probabilistic::{
    BayesianMultiOutputConfig, BayesianMultiOutputModel, BayesianMultiOutputModelTrained,
    GaussianProcessMultiOutput, GaussianProcessMultiOutputTrained, InferenceMethod, KernelFunction,
    PosteriorDistribution, PredictionWithUncertainty, PriorDistribution,
};

// Re-export ranking algorithms
pub use ranking::{
    BinaryClassifierModel, IndependentLabelPrediction, IndependentLabelPredictionTrained,
    ThresholdStrategy as RankingThresholdStrategy,
};

// Re-export sparse storage algorithms
pub use sparse_storage::{
    sparse_utils, CSRMatrix, MemoryUsage, SparseMultiOutput, SparseMultiOutputTrained,
    SparsityAnalysis, StorageRecommendation,
};

// Re-export multi-label algorithms
pub use multi_label::{
    BinaryRelevance, BinaryRelevanceTrained, LabelPowerset, LabelPowersetTrained,
    OneVsRestClassifier, OneVsRestClassifierTrained, PrunedLabelPowerset,
    PrunedLabelPowersetTrained, PruningStrategy,
};

// Re-export tree-based algorithms
pub use tree::{
    ClassificationCriterion, DAGInferenceMethod, MultiTargetDecisionTreeClassifier,
    MultiTargetDecisionTreeClassifierTrained, MultiTargetRegressionTree,
    MultiTargetRegressionTreeTrained, RandomForestMultiOutput, RandomForestMultiOutputTrained,
    TreeStructuredPredictor, TreeStructuredPredictorTrained,
};

// Re-export instance-based learning algorithms
pub use neighbors::{IBLRTrained, WeightFunction, IBLR};

// Re-export SVM algorithms
pub use svm::{
    MLTSVMTrained, MultiOutputSVM, MultiOutputSVMTrained, RankSVM, RankSVMTrained, RankingSVMModel,
    SVMKernel, SVMModel, ThresholdStrategy as SVMThresholdStrategy, TwinSVMModel, MLTSVM,
};

// Re-export sequence/structured prediction algorithms
pub use sequence::{
    FeatureFunction, FeatureType, HiddenMarkovModel, HiddenMarkovModelTrained,
    MaximumEntropyMarkovModel, MaximumEntropyMarkovModelTrained, StructuredPerceptron,
    StructuredPerceptronTrained,
};

// Re-export hierarchical classification and graph neural network algorithms
pub use hierarchical::{
    AggregationFunction, ConsistencyEnforcement, CostSensitiveHierarchicalClassifier,
    CostSensitiveHierarchicalClassifierTrained, CostStrategy, GraphNeuralNetwork,
    GraphNeuralNetworkTrained, MessagePassingVariant, OntologyAwareClassifier,
    OntologyAwareClassifierTrained,
};

// Re-export multi-label classification algorithms
pub use classification::{
    CalibratedBinaryRelevance, CalibratedBinaryRelevanceTrained, CalibrationMethod, CostMatrix,
    CostSensitiveBinaryRelevance, CostSensitiveBinaryRelevanceTrained, DistanceMetric, MLkNN,
    MLkNNTrained, RandomLabelCombinations, SimpleBinaryModel,
};

// Re-export comprehensive metrics and statistical testing functionality
pub use metrics::{
    average_precision_score,
    confidence_interval,
    coverage_error,
    f1_score,
    // Basic multi-label metrics
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision,
    // Statistical significance testing
    mcnemar_test,
    one_error,
    paired_t_test,
    // Per-label performance metrics
    per_label_metrics,
    precision_score_micro,
    ranking_loss,
    recall_score_micro,

    subset_accuracy,
    wilcoxon_signed_rank_test,
    ConfidenceInterval,
    PerLabelMetrics,

    StatisticalTestResult,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use crate::utilities::CLARE;
    use scirs2_core::ndarray::{array, Array2};
    use sklears_core::traits::{Fit, Predict};
    use sklears_core::types::Float;

    #[test]
    fn test_multi_output_classifier() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
        let y = array![[0, 1], [1, 0], [1, 1], [0, 0]];

        let moc = MultiOutputClassifier::new();
        let fitted = moc.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_targets(), 2);
        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&X.view()).unwrap();
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
        let fitted = mor.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_targets(), 2);

        let predictions = fitted.predict(&X.view()).unwrap();
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
        let fitted = moc.fit(&X.view(), &y).unwrap();

        // Test with wrong number of features
        let X_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(fitted.predict(&X_wrong.view()).is_err());
    }

    #[test]
    fn test_classifier_chain() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]];
        let y = array![[0, 1], [1, 0], [1, 1], [0, 0]];

        let cc = ClassifierChain::new();
        let fitted = cc.fit_simple(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_targets(), 2);
        assert_eq!(fitted.chain_order(), &[0, 1]); // Default order

        let predictions = fitted.predict_simple(&X.view()).unwrap();
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
        let fitted = cc.fit_simple(&X.view(), &y).unwrap();

        assert_eq!(fitted.chain_order(), &[1, 0]);

        let predictions = fitted.predict_simple(&X.view()).unwrap();
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
        let fitted = cc.fit_simple(&X.view(), &y).unwrap();

        // Test Monte Carlo predictions with probabilities
        let mc_probs = fitted
            .predict_monte_carlo(&X.view(), 100, Some(42))
            .unwrap();
        assert_eq!(mc_probs.dim(), (4, 2));

        // All probabilities should be between 0 and 1
        for prob in mc_probs.iter() {
            assert!(*prob >= 0.0 && *prob <= 1.0);
        }

        // Test Monte Carlo predictions with labels
        let mc_labels = fitted
            .predict_monte_carlo_labels(&X.view(), 100, Some(42))
            .unwrap();
        assert_eq!(mc_labels.dim(), (4, 2));

        // All predictions should be binary (0 or 1)
        for pred in mc_labels.iter() {
            assert!(*pred == 0 || *pred == 1);
        }

        // Test reproducibility with same random state
        let mc_probs2 = fitted
            .predict_monte_carlo(&X.view(), 100, Some(42))
            .unwrap();
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
        let fitted = cc.fit_simple(&X.view(), &y).unwrap();

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
        let fitted = rc.fit_simple(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_targets(), 2);
        assert_eq!(fitted.chain_order(), &[0, 1]); // Default order

        let predictions = fitted.predict_simple(&X.view()).unwrap();
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
        let fitted = rc.fit_simple(&X.view(), &y).unwrap();

        assert_eq!(fitted.chain_order(), &[1, 0]);

        let predictions = fitted.predict_simple(&X.view()).unwrap();
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
        let fitted = br.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_labels(), 2);
        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 2));

        // Check that predictions are binary (0 or 1)
        for pred in predictions.iter() {
            assert!(*pred == 0 || *pred == 1);
        }

        // Test probability predictions
        let probabilities = fitted.predict_proba(&X.view()).unwrap();
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
        let fitted = br.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_labels(), 1);

        let predictions = fitted.predict(&X.view()).unwrap();
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
        let fitted = br.fit(&X.view(), &y).unwrap();

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
        let fitted = lp.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_labels(), 2);
        assert_eq!(fitted.n_classes(), 4); // 4 unique combinations: [1,0], [0,1], [1,1], [0,0]

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 2));

        // Check that predictions are binary (0 or 1)
        for pred in predictions.iter() {
            assert!(*pred == 0 || *pred == 1);
        }

        // Test decision function
        let scores = fitted.decision_function(&X.view()).unwrap();
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
        let fitted = lp.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_labels(), 2);
        assert_eq!(fitted.n_classes(), 2); // Only 2 unique combinations: [1,0], [0,1]

        let predictions = fitted.predict(&X.view()).unwrap();
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
        let fitted = lp.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_labels(), 1);
        assert_eq!(fitted.n_classes(), 2); // 2 unique combinations: [1], [0]

        let predictions = fitted.predict(&X.view()).unwrap();
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
        let fitted = lp.fit(&X.view(), &y).unwrap();

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
        let fitted = lp.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_labels(), 2);
        assert_eq!(fitted.n_classes(), 1); // Only 1 unique combination

        let predictions = fitted.predict(&X.view()).unwrap();
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

        let fitted = plp.fit(&X.view(), &y).unwrap();

        // Should have pruned to only frequent combinations
        assert!(fitted.n_frequent_classes() <= 4); // At most [1,0], [0,1], [1,1], [0,0]
        assert_eq!(fitted.min_frequency(), 2);

        let predictions = fitted.predict(&X.view()).unwrap();
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

        let fitted = plp.fit(&X.view(), &y).unwrap();

        // Should have only 2 frequent combinations: [1,0] and [0,1]
        assert_eq!(fitted.n_frequent_classes(), 2);

        // The rare combination [1,1] should be mapped to one of the frequent ones
        let mapping = fitted.combination_mapping();
        let rare_combo = vec![1, 1];
        assert!(mapping.contains_key(&rare_combo));

        // The mapped combination should be one of the frequent ones
        let mapped = mapping.get(&rare_combo).unwrap();
        assert!(mapped == &vec![1, 0] || mapped == &vec![0, 1]);

        let predictions = fitted.predict(&X.view()).unwrap();
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
        let plp =
            PrunedLabelPowerset::new().strategy(PruningStrategy::DefaultMapping(vec![0, 1, 0])); // 3 elements for 2 labels
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
        let fitted = plp.fit(&X.view(), &y).unwrap();

        // Should have at least 1 combination, possibly 2 if default is added
        assert!(fitted.n_frequent_classes() >= 1);
        assert!(fitted.frequent_combinations().len() >= 1);

        // The frequent combinations should include [1, 0]
        assert!(fitted.frequent_combinations().contains(&vec![1, 0]));

        let predictions = fitted.predict(&X.view()).unwrap();
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

        let loss = metrics::hamming_loss(&y_true.view(), &y_pred.view()).unwrap();
        assert!((loss - 3.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_subset_accuracy() {
        let y_true = array![[1, 0, 1], [0, 1, 0], [1, 1, 1]];
        let y_pred = array![[1, 0, 1], [0, 1, 1], [1, 0, 1]]; // Only first subset matches

        let accuracy = metrics::subset_accuracy(&y_true.view(), &y_pred.view()).unwrap();
        assert!((accuracy - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_jaccard_score() {
        let y_true = array![[1, 0, 1], [0, 1, 0]];
        let y_pred = array![[1, 0, 0], [0, 1, 1]];

        let score = metrics::jaccard_score(&y_true.view(), &y_pred.view()).unwrap();
        // Sample 1: intersection=1, union=2, jaccard=0.5
        // Sample 2: intersection=1, union=2, jaccard=0.5
        // Average: 0.5
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_f1_score_micro() {
        let y_true = array![[1, 0, 1], [0, 1, 0], [1, 1, 1]];
        let y_pred = array![[1, 0, 0], [0, 1, 1], [1, 0, 1]];

        let f1 = metrics::f1_score(&y_true.view(), &y_pred.view(), "micro").unwrap();
        // TP=4, FP=1, FN=2
        // Precision = 4/5 = 0.8, Recall = 4/6 = 0.6667
        // F1 = 2 * 0.8 * 0.6667 / (0.8 + 0.6667) = 0.727
        assert!((f1 - 0.7272727272727273).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_f1_score_macro() {
        let y_true = array![[1, 0], [0, 1], [1, 1]];
        let y_pred = array![[1, 0], [0, 1], [1, 0]]; // Perfect for label 0, imperfect for label 1

        let f1 = metrics::f1_score(&y_true.view(), &y_pred.view(), "macro").unwrap();
        // Label 0: TP=2, FP=0, FN=0 -> F1=1.0
        // Label 1: TP=1, FP=0, FN=1 -> Precision=1.0, Recall=0.5, F1=0.667
        // Macro average: (1.0 + 0.667) / 2 = 0.833
        assert!((f1 - 0.8333333333333334).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_f1_score_samples() {
        let y_true = array![[1, 0], [0, 1], [1, 1]];
        let y_pred = array![[1, 0], [0, 1], [1, 0]];

        let f1 = metrics::f1_score(&y_true.view(), &y_pred.view(), "samples").unwrap();
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

        let coverage = metrics::coverage_error(&y_true.view(), &y_scores.view()).unwrap();
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

        let lrap =
            metrics::label_ranking_average_precision(&y_true.view(), &y_scores.view()).unwrap();
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
        let fitted = eoc.fit_simple(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_chains(), 3);
        assert_eq!(fitted.n_targets(), 2);

        let predictions = fitted.predict_simple(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 2));

        // Check that predictions are binary (0 or 1)
        for pred in predictions.iter() {
            assert!(*pred == 0 || *pred == 1);
        }

        // Test probability predictions
        let probabilities = fitted.predict_proba_simple(&X.view()).unwrap();
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
        let fitted = eoc.fit_simple(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_chains(), 1);

        let predictions = fitted.predict_simple(&X.view()).unwrap();
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
        let fitted = ovr.fit(&X.view(), &y).unwrap();

        assert_eq!(fitted.n_labels(), 2);
        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 2));

        // Check that predictions are binary (0 or 1)
        for pred in predictions.iter() {
            assert!(*pred == 0 || *pred == 1);
        }

        // Test probability predictions
        let probabilities = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probabilities.dim(), (4, 2));

        // Check that probabilities are in [0, 1]
        for prob in probabilities.iter() {
            assert!(*prob >= 0.0 && *prob <= 1.0);
        }

        // Test decision function
        let scores = fitted.decision_function(&X.view()).unwrap();
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

        let one_err = metrics::one_error(&y_true.view(), &y_scores.view()).unwrap();
        // All top-ranked labels are correct, so one-error should be 0
        assert!((one_err - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_one_error_with_errors() {
        let y_true = array![[1, 0], [0, 1]];
        let y_scores = array![[0.3, 0.7], [0.6, 0.4]]; // Top predictions are wrong

        let one_err = metrics::one_error(&y_true.view(), &y_scores.view()).unwrap();
        // Both samples have incorrect top predictions, so one-error should be 1.0
        assert!((one_err - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_ranking_loss() {
        let y_true = array![[1, 0], [0, 1]];
        let y_scores = array![[0.8, 0.2], [0.3, 0.7]]; // Correct ordering

        let ranking_loss = metrics::ranking_loss(&y_true.view(), &y_scores.view()).unwrap();
        // Perfect ranking, so loss should be 0
        assert!((ranking_loss - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_ranking_loss_with_errors() {
        let y_true = array![[1, 0], [0, 1]];
        let y_scores = array![[0.2, 0.8], [0.7, 0.3]]; // Incorrect ordering

        let ranking_loss = metrics::ranking_loss(&y_true.view(), &y_scores.view()).unwrap();
        // All pairs are incorrectly ordered, so loss should be 1.0
        assert!((ranking_loss - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_average_precision_score() {
        let y_true = array![[1, 0, 1], [0, 1, 0]];
        let y_scores = array![[0.9, 0.1, 0.8], [0.2, 0.9, 0.3]];

        let ap_score = metrics::average_precision_score(&y_true.view(), &y_scores.view()).unwrap();
        // With perfect ranking for both samples, AP should be 1.0
        assert!((ap_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_precision_recall_micro() {
        let y_true = array![[1, 0, 1], [0, 1, 0], [1, 1, 1]];
        let y_pred = array![[1, 0, 0], [0, 1, 1], [1, 0, 1]];

        let precision = metrics::precision_score_micro(&y_true.view(), &y_pred.view()).unwrap();
        let recall = metrics::recall_score_micro(&y_true.view(), &y_pred.view()).unwrap();

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

    #[test]
    fn test_label_analysis_basic() {
        let y = array![
            [1, 0, 1], // Cardinality 2
            [0, 1, 0], // Cardinality 1
            [1, 0, 1], // Cardinality 2 (duplicate)
            [0, 0, 0], // Cardinality 0
            [1, 1, 1], // Cardinality 3
        ];

        let results = label_analysis::analyze_combinations(&y.view()).unwrap();

        assert_eq!(results.total_samples, 5);
        assert_eq!(results.combinations[0].combination.len(), 3); // Number of labels = 3
        assert_eq!(results.unique_combinations, 4); // [1,0,1], [0,1,0], [0,0,0], [1,1,1]

        // Check that [1,0,1] is most frequent (appears twice)
        assert_eq!(
            results.most_frequent.as_ref().unwrap().combination,
            vec![1, 0, 1]
        );
        assert_eq!(results.most_frequent.as_ref().unwrap().frequency, 2);
        assert!((results.most_frequent.as_ref().unwrap().relative_frequency - 0.4).abs() < 1e-10);
        assert_eq!(results.most_frequent.as_ref().unwrap().cardinality, 2);

        // Average cardinality should be (2+1+2+0+3)/5 = 1.6
        assert!((results.average_cardinality - 1.6).abs() < 1e-10);
    }

    #[test]
    fn test_label_analysis_utility_functions() {
        let y = array![
            [1, 0],
            [1, 0],
            [1, 0], // Frequent: [1, 0] appears 3 times
            [0, 1],
            [0, 1], // Frequent: [0, 1] appears 2 times
            [1, 1]  // Rare: [1, 1] appears 1 time
        ];

        let results = label_analysis::analyze_combinations(&y.view()).unwrap();

        // Test get_rare_combinations
        let rare = label_analysis::get_rare_combinations(&results, 2);
        assert_eq!(rare.len(), 2); // [0,1] freq=2 and [1,1] freq=1 are both <= threshold
                                   // Find the combination with frequency 1
        let freq_1_combo = rare.iter().find(|combo| combo.frequency == 1).unwrap();
        assert_eq!(freq_1_combo.combination, vec![1, 1]);

        // Test get_combinations_by_cardinality
        let cardinality_1 = label_analysis::get_combinations_by_cardinality(&results, 1);
        assert_eq!(cardinality_1.len(), 2); // [1, 0] and [0, 1]

        let cardinality_2 = label_analysis::get_combinations_by_cardinality(&results, 2);
        assert_eq!(cardinality_2.len(), 1); // [1, 1]
        assert_eq!(cardinality_2[0].combination, vec![1, 1]);
    }

    #[test]
    fn test_label_cooccurrence_matrix() {
        let y = array![
            [1, 1, 0], // Labels 0 and 1 co-occur
            [1, 0, 1], // Labels 0 and 2 co-occur
            [0, 1, 1], // Labels 1 and 2 co-occur
            [1, 1, 1], // All labels co-occur
        ];

        let cooccurrence = label_analysis::label_cooccurrence_matrix(&y.view()).unwrap();
        assert_eq!(cooccurrence.dim(), (3, 3));

        // Label 0 appears with itself in samples 0, 1, 3 = 3 times
        assert_eq!(cooccurrence[[0, 0]], 3);
        // Label 1 appears with itself in samples 0, 2, 3 = 3 times
        assert_eq!(cooccurrence[[1, 1]], 3);
        // Label 2 appears with itself in samples 1, 2, 3 = 3 times
        assert_eq!(cooccurrence[[2, 2]], 3);

        // Labels 0 and 1 co-occur in samples 0, 3 = 2 times
        assert_eq!(cooccurrence[[0, 1]], 2);
        assert_eq!(cooccurrence[[1, 0]], 2);

        // Labels 0 and 2 co-occur in samples 1, 3 = 2 times
        assert_eq!(cooccurrence[[0, 2]], 2);
        assert_eq!(cooccurrence[[2, 0]], 2);

        // Labels 1 and 2 co-occur in samples 2, 3 = 2 times
        assert_eq!(cooccurrence[[1, 2]], 2);
        assert_eq!(cooccurrence[[2, 1]], 2);
    }

    #[test]
    fn test_label_correlation_matrix() {
        let y = array![[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0],];

        let correlation = label_analysis::label_correlation_matrix(&y.view()).unwrap();
        assert_eq!(correlation.dim(), (3, 3));

        // Diagonal should be 1.0 (perfect self-correlation)
        assert!((correlation[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((correlation[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((correlation[[2, 2]] - 1.0).abs() < 1e-10);

        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((correlation[[i, j]] - correlation[[j, i]]).abs() < 1e-10);
            }
        }

        // All correlations should be between -1 and 1
        for i in 0..3 {
            for j in 0..3 {
                assert!(correlation[[i, j]] >= -1.0 && correlation[[i, j]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_label_analysis_invalid_input() {
        // Test with non-binary labels
        let y_bad = array![[2, 1], [1, 0]]; // Contains non-binary value
        assert!(label_analysis::analyze_combinations(&y_bad.view()).is_err());

        // Test with empty array
        let y_empty = Array2::<i32>::zeros((0, 2));
        assert!(label_analysis::analyze_combinations(&y_empty.view()).is_err());

        let y_no_labels = Array2::<i32>::zeros((2, 0));
        assert!(label_analysis::analyze_combinations(&y_no_labels.view()).is_err());

        // Test cooccurrence matrix with empty data
        assert!(label_analysis::label_cooccurrence_matrix(&y_empty.view()).is_err());
        assert!(label_analysis::label_correlation_matrix(&y_empty.view()).is_err());
    }

    #[test]
    fn test_label_analysis_edge_cases() {
        // Test with single sample
        let y_single = array![[1, 0, 1]];
        let results = label_analysis::analyze_combinations(&y_single.view()).unwrap();

        assert_eq!(results.total_samples, 1);
        assert_eq!(results.unique_combinations, 1);
        assert_eq!(
            results.most_frequent.as_ref().unwrap().combination,
            vec![1, 0, 1]
        );
        assert_eq!(
            results.least_frequent.as_ref().unwrap().combination,
            vec![1, 0, 1]
        );
        assert_eq!(results.average_cardinality, 2.0);

        // Test with all zeros
        let y_zeros = array![[0, 0], [0, 0]];
        let results = label_analysis::analyze_combinations(&y_zeros.view()).unwrap();

        assert_eq!(results.average_cardinality, 0.0);

        // Test with all ones
        let y_ones = array![[1, 1], [1, 1]];
        let results = label_analysis::analyze_combinations(&y_ones.view()).unwrap();

        assert_eq!(results.average_cardinality, 2.0);
    }

    #[test]
    fn test_iblr_basic_functionality() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label classification targets

        let iblr = IBLR::new().k_neighbors(2);
        let trained_iblr = iblr.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained_iblr.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (4, 2));

        // Check that predictions are binary (0 or 1)
        for i in 0..4 {
            for j in 0..2 {
                assert!(predictions[[i, j]] == 0 || predictions[[i, j]] == 1);
            }
        }
    }

    #[test]
    fn test_iblr_configuration() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![[1, 0], [0, 1], [1, 1]]; // Multi-label classification targets

        // Test different k values
        let iblr1 = IBLR::new().k_neighbors(1);
        let iblr2 = IBLR::new().k_neighbors(2); // Must be < n_samples (3)

        let trained1 = iblr1.fit(&X.view(), &y.view()).unwrap();
        let trained2 = iblr2.fit(&X.view(), &y.view()).unwrap();

        let pred1 = trained1.predict(&X.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();

        assert_eq!(pred1.dim(), (3, 2));
        assert_eq!(pred2.dim(), (3, 2));

        // Test weight functions
        let iblr_uniform = IBLR::new().k_neighbors(2).weights(WeightFunction::Uniform);
        let iblr_distance = IBLR::new().k_neighbors(2).weights(WeightFunction::Distance);

        let trained_uniform = iblr_uniform.fit(&X.view(), &y.view()).unwrap();
        let trained_distance = iblr_distance.fit(&X.view(), &y.view()).unwrap();

        let pred_uniform = trained_uniform.predict(&X.view()).unwrap();
        let pred_distance = trained_distance.predict(&X.view()).unwrap();

        assert_eq!(pred_uniform.dim(), (3, 2));
        assert_eq!(pred_distance.dim(), (3, 2));
    }

    #[test]
    fn test_iblr_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1, 0], [0, 1], [1, 1]]; // Mismatched samples (3 vs 2)

        let iblr = IBLR::new();
        assert!(iblr.fit(&X.view(), &y.view()).is_err());

        // Test k_neighbors validation
        let y_valid = array![[1, 0], [0, 1]]; // Matching samples

        let iblr_zero_k = IBLR::new().k_neighbors(0);
        assert!(iblr_zero_k.fit(&X.view(), &y_valid.view()).is_err());

        let iblr_large_k = IBLR::new().k_neighbors(5); // More than samples
        assert!(iblr_large_k.fit(&X.view(), &y_valid.view()).is_err());

        // Test prediction with wrong feature dimensions
        let X_train = array![[1.0, 2.0], [2.0, 3.0]];
        let y_train = array![[1, 0], [0, 1]];
        let iblr_for_predict = IBLR::new().k_neighbors(1); // Must be < n_samples (2)
        let trained = iblr_for_predict
            .fit(&X_train.view(), &y_train.view())
            .unwrap();

        let X_wrong_features = array![[1.0, 2.0, 3.0]]; // Extra feature
        assert!(trained.predict(&X_wrong_features.view()).is_err());

        // Test empty data
        let X_empty = Array2::<Float>::zeros((0, 2));
        let y_empty = Array2::<i32>::zeros((0, 2));
        assert!(IBLR::new().fit(&X_empty.view(), &y_empty.view()).is_err());
    }

    #[test]
    fn test_iblr_weight_functions() {
        let X = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let y = array![[1, 1], [0, 1], [1, 0], [0, 0]]; // Binary classification labels

        // Test uniform weighting
        let iblr_uniform = IBLR::new().k_neighbors(3).weights(WeightFunction::Uniform);
        let trained_uniform = iblr_uniform.fit(&X.view(), &y.view()).unwrap();
        let pred_uniform = trained_uniform.predict(&X.view()).unwrap();

        // Test distance weighting
        let iblr_distance = IBLR::new().k_neighbors(3).weights(WeightFunction::Distance);
        let trained_distance = iblr_distance.fit(&X.view(), &y.view()).unwrap();
        let pred_distance = trained_distance.predict(&X.view()).unwrap();

        // Predictions should be reasonable for both
        assert_eq!(pred_uniform.dim(), (4, 2));
        assert_eq!(pred_distance.dim(), (4, 2));

        // Check that all predictions are binary (0 or 1)
        for i in 0..4 {
            for j in 0..2 {
                assert!(pred_uniform[[i, j]] == 0 || pred_uniform[[i, j]] == 1);
                assert!(pred_distance[[i, j]] == 0 || pred_distance[[i, j]] == 1);
            }
        }
    }

    #[test]
    fn test_iblr_single_neighbor() {
        let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![[1, 0], [0, 1], [1, 1]]; // Binary classification labels

        let iblr = IBLR::new().k_neighbors(1);
        let trained = iblr.fit(&X.view(), &y.view()).unwrap();

        // Test prediction on training data (should be exact for k=1)
        let predictions = trained.predict(&X.view()).unwrap();

        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(predictions[[i, j]], y[[i, j]]);
            }
        }
    }

    #[test]
    fn test_iblr_interpolation() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![[0, 0], [1, 1], [0, 1]]; // Binary classification labels

        let iblr = IBLR::new().k_neighbors(2);
        let trained = iblr.fit(&X.view(), &y.view()).unwrap();

        // Test prediction at midpoint
        let X_test = array![[0.5, 0.5]];
        let prediction = trained.predict(&X_test.view()).unwrap();

        // Should predict binary values (0 or 1)
        assert!(prediction[[0, 0]] == 0 || prediction[[0, 0]] == 1);
        assert!(prediction[[0, 1]] == 0 || prediction[[0, 1]] == 1);
    }

    #[test]
    fn test_clare_basic_functionality() {
        let X = array![[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [5.5, 5.5]];
        let y = array![[1, 0], [1, 0], [0, 1], [0, 1]]; // Two clear clusters with different label patterns

        let clare = CLARE::new().n_clusters(2).random_state(42);
        let trained_clare = clare.fit(&X.view(), &y).unwrap();
        let predictions = trained_clare.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (4, 2));

        // Verify cluster centers and assignments were learned
        assert_eq!(trained_clare.cluster_centers().dim(), (2, 2));
        assert_eq!(trained_clare.cluster_assignments().len(), 4);
    }

    #[test]
    fn test_clare_configuration() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        // Test different configurations
        let clare1 = CLARE::new().n_clusters(2).threshold(0.3);
        let clare2 = CLARE::new().n_clusters(3).max_iter(50);
        let clare3 = CLARE::new().random_state(123);

        let trained1 = clare1.fit(&X.view(), &y).unwrap();
        let trained2 = clare2.fit(&X.view(), &y).unwrap();
        let trained3 = clare3.fit(&X.view(), &y).unwrap();

        let pred1 = trained1.predict(&X.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();
        let pred3 = trained3.predict(&X.view()).unwrap();

        assert_eq!(pred1.dim(), (4, 2));
        assert_eq!(pred2.dim(), (4, 2));
        assert_eq!(pred3.dim(), (4, 2));

        // Test accessors
        assert_eq!(trained1.threshold(), 0.3);
        assert_eq!(trained1.cluster_centers().dim(), (2, 2));
        assert_eq!(trained2.cluster_centers().dim(), (3, 2));
    }

    #[test]
    fn test_clare_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1, 0], [0, 1], [1, 1]]; // Mismatched samples

        let clare = CLARE::new();
        assert!(clare.fit(&X.view(), &y).is_err());

        // Test n_clusters validation
        let y_valid = array![[1, 0], [0, 1]];

        let clare_zero_clusters = CLARE::new().n_clusters(0);
        assert!(clare_zero_clusters.fit(&X.view(), &y_valid).is_err());

        let clare_too_many_clusters = CLARE::new().n_clusters(5); // More than samples
        assert!(clare_too_many_clusters.fit(&X.view(), &y_valid).is_err());

        // Test non-binary labels
        let y_non_binary = array![[1, 2], [0, 1]]; // Contains 2
        assert!(CLARE::new().fit(&X.view(), &y_non_binary).is_err());

        // Test prediction with wrong feature dimensions
        let X_train = array![[1.0, 2.0], [2.0, 3.0]];
        let y_train = array![[1, 0], [0, 1]];
        let clare_for_predict = CLARE::new().n_clusters(2);
        let trained = clare_for_predict.fit(&X_train.view(), &y_train).unwrap();

        let X_wrong_features = array![[1.0, 2.0, 3.0]]; // Extra feature
        assert!(trained.predict(&X_wrong_features.view()).is_err());

        // Test empty data
        let X_empty = Array2::<Float>::zeros((0, 2));
        let y_empty = Array2::<i32>::zeros((0, 2));
        assert!(CLARE::new().fit(&X_empty.view(), &y_empty).is_err());
    }

    #[test]
    fn test_clare_threshold_prediction() {
        let X = array![[1.0, 1.0], [1.2, 1.2], [5.0, 5.0], [5.2, 5.2]];
        let y = array![[1, 0], [1, 0], [0, 1], [0, 1]];

        let clare = CLARE::new().n_clusters(2).threshold(0.3).random_state(42);
        let trained_clare = clare.fit(&X.view(), &y).unwrap();

        // Test predictions are binary
        let predictions = trained_clare.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 2));

        // All predictions should be 0 or 1
        for pred in predictions.iter() {
            assert!(*pred == 0 || *pred == 1);
        }

        // Verify threshold was set correctly
        assert_eq!(trained_clare.threshold(), 0.3);
    }

    #[test]
    fn test_clare_clustering_consistency() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1], // First cluster
            [5.0, 5.0],
            [5.1, 5.1] // Second cluster
        ];
        let y = array![
            [1, 0],
            [1, 0], // First cluster: always label 0 active
            [0, 1],
            [0, 1] // Second cluster: always label 1 active
        ];

        let clare = CLARE::new().n_clusters(2).threshold(0.5).random_state(42);
        let trained_clare = clare.fit(&X.view(), &y).unwrap();
        let predictions = trained_clare.predict(&X.view()).unwrap();

        // With clear clustering, predictions should match patterns
        assert_eq!(predictions.dim(), (4, 2));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));

        // Test threshold accessor
        assert!((trained_clare.threshold() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_clare_single_cluster() {
        let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![[1, 0], [0, 1], [1, 1]];

        // Use only 1 cluster
        let clare = CLARE::new().n_clusters(1);
        let trained_clare = clare.fit(&X.view(), &y).unwrap();
        let predictions = trained_clare.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (3, 2));
        assert_eq!(trained_clare.cluster_centers().dim(), (1, 2));

        // With 1 cluster, all samples should get same prediction
        // (based on average label frequency)
        for i in 1..3 {
            for j in 0..2 {
                assert_eq!(predictions[[0, j]], predictions[[i, j]]);
            }
        }
    }

    #[test]
    fn test_clare_reproducibility() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        // Train two models with same random state
        let clare1 = CLARE::new().n_clusters(2).random_state(42);
        let trained1 = clare1.fit(&X.view(), &y).unwrap();

        let clare2 = CLARE::new().n_clusters(2).random_state(42);
        let trained2 = clare2.fit(&X.view(), &y).unwrap();

        // Should produce same cluster centers
        let centers1 = trained1.cluster_centers();
        let centers2 = trained2.cluster_centers();

        for i in 0..centers1.nrows() {
            for j in 0..centers1.ncols() {
                assert!((centers1[[i, j]] - centers2[[i, j]]).abs() < 1e-10);
            }
        }

        // Should produce same predictions
        let pred1 = trained1.predict(&X.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();

        for i in 0..pred1.nrows() {
            for j in 0..pred1.ncols() {
                assert_eq!(pred1[[i, j]], pred2[[i, j]]);
            }
        }
    }

    #[test]
    fn test_mltsvm_basic_functionality() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label binary

        let mltsvm = MLTSVM::new().c1(1.0).c2(1.0);
        let trained_mltsvm = mltsvm.fit(&X.view(), &y).unwrap();
        let predictions = trained_mltsvm.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (4, 2));
        assert_eq!(trained_mltsvm.n_labels(), 2);

        // All predictions should be binary (0 or 1)
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn test_mltsvm_configuration() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        // Test different configurations
        let mltsvm1 = MLTSVM::new().c1(0.5).c2(1.5);
        let mltsvm2 = MLTSVM::new().epsilon(1e-8).max_iter(500);

        let trained1 = mltsvm1.fit(&X.view(), &y).unwrap();
        let trained2 = mltsvm2.fit(&X.view(), &y).unwrap();

        let pred1 = trained1.predict(&X.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();

        assert_eq!(pred1.dim(), (4, 2));
        assert_eq!(pred2.dim(), (4, 2));
    }

    #[test]
    fn test_mltsvm_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1, 0], [0, 1], [1, 1]]; // Mismatched samples

        let mltsvm = MLTSVM::new();
        assert!(mltsvm.fit(&X.view(), &y).is_err());

        // Test non-binary labels
        let y_non_binary = array![[1, 2], [0, 1]]; // Contains 2
        assert!(MLTSVM::new().fit(&X.view(), &y_non_binary).is_err());

        // Test prediction with wrong feature dimensions
        let X_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let y_train = array![[1, 0], [0, 1], [1, 1], [0, 0]];
        let mltsvm_for_predict = MLTSVM::new();
        let trained = mltsvm_for_predict.fit(&X_train.view(), &y_train).unwrap();

        let X_wrong_features = array![[1.0, 2.0, 3.0]]; // Extra feature
        assert!(trained.predict(&X_wrong_features.view()).is_err());

        // Test empty data
        let X_empty = Array2::<Float>::zeros((0, 2));
        let y_empty = Array2::<i32>::zeros((0, 2));
        assert!(MLTSVM::new().fit(&X_empty.view(), &y_empty).is_err());
    }

    #[test]
    fn test_mltsvm_decision_function() {
        let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![[1, 0], [1, 0], [0, 1], [0, 1]];

        let mltsvm = MLTSVM::new();
        let trained_mltsvm = mltsvm.fit(&X.view(), &y).unwrap();

        // Test decision function
        let decision_values = trained_mltsvm.decision_function(&X.view()).unwrap();
        assert_eq!(decision_values.dim(), (4, 2));

        // Decision values should be real numbers (no constraints on range)
        // Just check that we get reasonable outputs
        for &val in decision_values.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_mltsvm_separable_data() {
        let X = array![
            [0.0, 0.0],
            [0.5, 0.5], // Negative class cluster
            [3.0, 3.0],
            [3.5, 3.5] // Positive class cluster
        ];
        let y = array![
            [0, 1],
            [0, 1], // First label: negative, Second label: positive
            [1, 0],
            [1, 0] // First label: positive, Second label: negative
        ];

        let mltsvm = MLTSVM::new().c1(1.0).c2(1.0);
        let trained_mltsvm = mltsvm.fit(&X.view(), &y).unwrap();
        let predictions = trained_mltsvm.predict(&X.view()).unwrap();

        // With linearly separable data, MLTSVM should perform well
        let mut correct_predictions = 0;
        let total_predictions = predictions.len();

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                if predictions[[i, j]] == y[[i, j]] {
                    correct_predictions += 1;
                }
            }
        }

        let accuracy = correct_predictions as Float / total_predictions as Float;
        // Should get reasonably good accuracy on separable data
        assert!(accuracy >= 0.5); // At least better than random
    }

    #[test]
    fn test_mltsvm_feature_scaling() {
        // Test with features of very different scales
        let X = array![
            [1000.0, 0.001],
            [2000.0, 0.002],
            [3000.0, 0.003],
            [4000.0, 0.004]
        ];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        let mltsvm = MLTSVM::new();
        let trained_mltsvm = mltsvm.fit(&X.view(), &y).unwrap();
        let predictions = trained_mltsvm.predict(&X.view()).unwrap();

        // Should handle feature scaling internally
        assert_eq!(predictions.dim(), (4, 2));

        // All predictions should be binary
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn test_mltsvm_consistency() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        // Train the same model multiple times (deterministic should give same results)
        let mltsvm1 = MLTSVM::new().c1(1.0).c2(1.0);
        let trained1 = mltsvm1.fit(&X.view(), &y).unwrap();

        let mltsvm2 = MLTSVM::new().c1(1.0).c2(1.0);
        let trained2 = mltsvm2.fit(&X.view(), &y).unwrap();

        let pred1 = trained1.predict(&X.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();

        // Should be deterministic (same predictions)
        for i in 0..pred1.nrows() {
            for j in 0..pred1.ncols() {
                assert_eq!(pred1[[i, j]], pred2[[i, j]]);
            }
        }
    }

    #[test]
    fn test_ranksvm_basic_functionality() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label binary

        let ranksvm = RankSVM::new().c(1.0);
        let trained_ranksvm = ranksvm.fit(&X.view(), &y).unwrap();
        let predictions = trained_ranksvm.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (4, 2));
        assert_eq!(trained_ranksvm.n_labels(), 2);

        // All predictions should be binary (0 or 1)
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn test_ranksvm_threshold_strategies() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        // Test different threshold strategies
        let ranksvm1 = RankSVM::new().threshold_strategy(SVMThresholdStrategy::Fixed(0.5));
        let ranksvm2 = RankSVM::new().threshold_strategy(SVMThresholdStrategy::OptimizeF1);
        let ranksvm3 = RankSVM::new().threshold_strategy(SVMThresholdStrategy::TopK(2));
        let ranksvm4 = RankSVM::new().threshold_strategy(SVMThresholdStrategy::OptimizeF1);

        let trained1 = ranksvm1.fit(&X.view(), &y).unwrap();
        let trained2 = ranksvm2.fit(&X.view(), &y).unwrap();
        let trained3 = ranksvm3.fit(&X.view(), &y).unwrap();
        let trained4 = ranksvm4.fit(&X.view(), &y).unwrap();

        let pred1 = trained1.predict(&X.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();
        let pred3 = trained3.predict(&X.view()).unwrap();
        let pred4 = trained4.predict(&X.view()).unwrap();

        assert_eq!(pred1.dim(), (4, 2));
        assert_eq!(pred2.dim(), (4, 2));
        assert_eq!(pred3.dim(), (4, 2));
        assert_eq!(pred4.dim(), (4, 2));

        // Test threshold accessors
        assert_eq!(trained1.thresholds().len(), 2);
        assert_eq!(trained2.thresholds().len(), 2);
        assert_eq!(trained3.thresholds().len(), 2);
    }

    #[test]
    fn test_ranksvm_decision_function() {
        let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![[1, 0], [1, 0], [0, 1], [0, 1]];

        let ranksvm = RankSVM::new();
        let trained_ranksvm = ranksvm.fit(&X.view(), &y).unwrap();

        // Test decision function
        let scores = trained_ranksvm.decision_function(&X.view()).unwrap();
        assert_eq!(scores.dim(), (4, 2));

        // Scores should be real numbers
        for &score in scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_ranksvm_predict_ranking() {
        let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![[1, 0, 0], [0, 1, 0], [0, 0, 1]]; // Three labels, one active per sample

        let ranksvm = RankSVM::new();
        let trained_ranksvm = ranksvm.fit(&X.view(), &y).unwrap();

        // Test ranking prediction
        let rankings = trained_ranksvm.predict_ranking(&X.view()).unwrap();

        assert_eq!(rankings.dim(), (3, 3)); // 3 samples, 3 labels
        for sample_idx in 0..3 {
            let mut ranking_vec = Vec::new();
            for label_idx in 0..3 {
                ranking_vec.push(rankings[[sample_idx, label_idx]]);
            }
            // Should contain all label indices
            let mut sorted_ranking = ranking_vec.clone();
            sorted_ranking.sort();
            assert_eq!(sorted_ranking, vec![0, 1, 2]);
        }
    }

    #[test]
    fn test_ranksvm_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1, 0], [0, 1], [1, 1]]; // Mismatched samples

        let ranksvm = RankSVM::new();
        assert!(ranksvm.fit(&X.view(), &y).is_err());

        // Test non-binary labels
        let y_non_binary = array![[1, 2], [0, 1]]; // Contains 2
        assert!(RankSVM::new().fit(&X.view(), &y_non_binary).is_err());

        // Test prediction with wrong feature dimensions
        let X_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let y_train = array![[1, 0], [0, 1], [1, 1], [0, 0]];
        let ranksvm_for_predict = RankSVM::new();
        let trained = ranksvm_for_predict.fit(&X_train.view(), &y_train).unwrap();

        let X_wrong_features = array![[1.0, 2.0, 3.0]]; // Extra feature
        assert!(trained.predict(&X_wrong_features.view()).is_err());
        assert!(trained.decision_function(&X_wrong_features.view()).is_err());
        assert!(trained.predict_ranking(&X_wrong_features.view()).is_err());

        // Test empty data
        let X_empty = Array2::<Float>::zeros((0, 2));
        let y_empty = Array2::<i32>::zeros((0, 2));
        assert!(RankSVM::new().fit(&X_empty.view(), &y_empty).is_err());
    }

    #[test]
    fn test_ranksvm_configuration() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        // Test different configurations
        let ranksvm1 = RankSVM::new().c(0.5).epsilon(1e-8);
        let ranksvm2 = RankSVM::new().max_iter(500);

        let trained1 = ranksvm1.fit(&X.view(), &y).unwrap();
        let trained2 = ranksvm2.fit(&X.view(), &y).unwrap();

        let pred1 = trained1.predict(&X.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();

        assert_eq!(pred1.dim(), (4, 2));
        assert_eq!(pred2.dim(), (4, 2));
    }

    #[test]
    fn test_ranksvm_ranking_consistency() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![
            [0, 0, 1], // Last label should rank highest
            [0, 1, 0], // Middle label should rank highest
            [1, 0, 0]  // First label should rank highest
        ];

        let ranksvm = RankSVM::new().threshold_strategy(SVMThresholdStrategy::OptimizeF1);
        let trained_ranksvm = ranksvm.fit(&X.view(), &y).unwrap();

        let rankings = trained_ranksvm.predict_ranking(&X.view()).unwrap();
        let scores = trained_ranksvm.decision_function(&X.view()).unwrap();

        // Check that rankings are consistent with scores
        for i in 0..3 {
            // First ranked label should have highest score
            let top_label = rankings[[i, 0]];
            for j in 1..3 {
                let other_label = rankings[[i, j]];
                assert!(scores[[i, top_label]] >= scores[[i, other_label]]);
            }
        }
    }

    #[test]
    fn test_ranksvm_single_class_handling() {
        let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];

        // Test with all positive for one label, mixed for other
        let y = array![[1, 0], [1, 1], [1, 0]]; // First label: all positive, second label: mixed

        let ranksvm = RankSVM::new();
        let trained = ranksvm.fit(&X.view(), &y).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        let scores = trained.decision_function(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (3, 2));
        assert_eq!(scores.dim(), (3, 2));

        // All scores should be finite
        for &score in scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_ranksvm_reproducibility() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        // Train two models with same configuration
        let ranksvm1 = RankSVM::new().c(1.0).epsilon(1e-6);
        let trained1 = ranksvm1.fit(&X.view(), &y).unwrap();

        let ranksvm2 = RankSVM::new().c(1.0).epsilon(1e-6);
        let trained2 = ranksvm2.fit(&X.view(), &y).unwrap();

        let pred1 = trained1.predict(&X.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();
        let scores1 = trained1.decision_function(&X.view()).unwrap();
        let scores2 = trained2.decision_function(&X.view()).unwrap();

        // Should be deterministic (same predictions and scores)
        for i in 0..pred1.nrows() {
            for j in 0..pred1.ncols() {
                assert_eq!(pred1[[i, j]], pred2[[i, j]]);
                assert!((scores1[[i, j]] - scores2[[i, j]]).abs() < 1e-10);
            }
        }
    }
}
