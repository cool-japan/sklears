#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
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
pub mod performance;
pub mod probabilistic;
pub mod ranking;
pub mod recurrent;
pub mod regularization;
pub mod sequence;
pub mod sparse_storage;
pub mod streaming;
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
    EnsembleBayesianConfig, EnsembleBayesianModel, EnsembleBayesianModelTrained, EnsembleStrategy,
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

// Re-export streaming and incremental learning algorithms
pub use streaming::{
    IncrementalMultiOutputRegression, IncrementalMultiOutputRegressionConfig,
    IncrementalMultiOutputRegressionTrained, StreamingMultiOutput, StreamingMultiOutputConfig,
    StreamingMultiOutputTrained,
};

// Re-export performance optimization algorithms
pub use performance::{
    EarlyStopping, EarlyStoppingConfig, PredictionCache, WarmStartRegressor,
    WarmStartRegressorConfig, WarmStartRegressorTrained,
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
mod tests_core;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_advanced;
