#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
//! Feature selection algorithms
//!
//! This module provides algorithms for selecting relevant features from data,
//! compatible with scikit-learn's feature_selection module.

pub mod automl;
pub mod base;
pub mod bayesian;
pub mod benchmark;
// TODO: ndarray 0.17 - uses disabled embedded module
// pub mod comparison_tests;
pub mod comprehensive_benchmark;
pub mod domain_benchmark;
pub mod domain_specific;
// TODO: ndarray 0.17 HRTB trait bound issues
// pub mod embedded;
// pub mod ensemble_selectors;
pub mod evaluation;
pub mod filter;
pub mod fluent_api;
// TODO: ndarray 0.17 HRTB trait bound issues
// pub mod genetic_optimization;
pub mod group_selection;
pub mod hierarchical;
pub mod ml_based;
pub mod multi_label;
pub mod optimization;
pub mod parallel;
pub mod performance;
pub mod pipeline;
pub mod plugin;
pub mod regularization_selectors;
#[cfg(feature = "serde")]
pub mod serialization;
pub mod spectral;
pub mod statistical_tests;
pub mod streaming;
// TODO: ndarray 0.17 HRTB trait bound issues
// pub mod tree_based_selectors;
pub mod type_safe;
// TODO: ndarray 0.17 HRTB trait bound issues
// pub mod validation;
// TODO: ndarray 0.17 HRTB trait bound issues - closures in generic methods
// pub mod wrapper;

pub use automl::{
    analyze_and_recommend, comprehensive_automl, quick_automl, AdvancedHyperparameterOptimizer,
    AutoMLBenchmark, AutoMLError, AutoMLFactory, AutoMLFactoryConfig, AutoMLMethod, AutoMLResults,
    AutoMLSummary, AutomatedFeatureSelectionPipeline, ComputationalBudget, DataAnalyzer,
    DataCharacteristics, HyperparameterOptimizer, MethodSelector, PipelineConfig,
    PipelineOptimizer, PreprocessingIntegration, TargetType, ValidationStrategy,
};
pub use base::*;
pub use benchmark::{
    BenchmarkConfig, BenchmarkDataset, BenchmarkSuiteResults, BenchmarkableMethod,
    FeatureSelectionBenchmark, RandomSelectionMethod, UnivariateFilterMethod,
};
pub use filter::*;

// Re-export commonly used items
pub use crate::filter::{
    CompressedSensingAlgorithm, CompressedSensingSelector, CorrelationThreshold,
    GenericUnivariateSelect, HighDimensionalInference, ImbalancedDataSelector, ImbalancedStrategy,
    InferenceMethod, KnockoffSelector, KnockoffType, RReliefF, Relief, ReliefF, SelectFdr,
    SelectFpr, SelectFwe, SelectKBest, SelectKBestParallel, SelectPercentile,
    SureIndependenceScreening, VarianceThreshold,
};

// TODO: ndarray 0.17 HRTB trait bound issues - closures in generic methods
// pub use crate::wrapper::{
//     FeatureImportance, HasCoefficients, IndexableTarget, RFECVResults, SelectFromModel,
//     SequentialFeatureSelector, RFE, RFECV,
// };

// TODO: ndarray 0.17 HRTB trait bound issues
// pub use crate::embedded::{
//     ConsensusFeatureSelector, ConsensusMethod, ConsensusThresholdParams, MultiTaskFeatureSelector,
//     StabilitySelector,
// };

// TODO: ndarray 0.17 HRTB trait bound issues
// pub use crate::genetic_optimization::{
//     CostSensitiveObjective, CrossValidator, FairnessAwareObjective, FairnessMetric,
//     FeatureCountObjective, FeatureDiversityObjective, FeatureImportanceObjective, GeneticSelector,
//     Individual, KFold, MultiObjectiveFeatureSelector, MultiObjectiveMethod, ObjectiveFunction,
//     PredictivePerformanceObjective,
// };

// TODO: ndarray 0.17 HRTB trait bound issues
// pub use crate::ensemble_selectors::{
//     extract_features, AggregationMethod, BootstrapSelector, BorutaSelector, EnsembleFeatureRanking,
//     SelectorFunction, UnivariateMethod, UnivariateSelector,
// };

// TODO: ndarray 0.17 HRTB trait bound issues
// pub use crate::tree_based_selectors::{
//     GradientBoostingSelector, TreeImportance, TreeImportanceSelector, TreeSelector,
// };

pub use crate::regularization_selectors::{ElasticNetSelector, LassoSelector, RidgeSelector};

pub use crate::domain_specific::{
    AdvancedNLPFeatureSelector, BioinformaticsFeatureSelector, FinanceFeatureSelector,
    GraphFeatureSelector, ImageFeatureSelector, MultiModalFeatureSelector, TextFeatureSelector,
    TimeSeriesSelector,
};

pub use crate::domain_benchmark::{
    run_quick_benchmark, BenchmarkConfig as DomainBenchmarkConfig, BenchmarkResult, BenchmarkSuite,
    BenchmarkSummary, DomainBenchmarkFramework,
};

pub use crate::ml_based::{
    AttentionFeatureSelector, MetaLearningFeatureSelector, NeuralFeatureSelector, RLFeatureSelector,
};

// Let's export only the items that actually exist
pub use crate::evaluation::{
    ComparativeAnalysis, FeatureInteractionAnalysis, FeatureSetDiversityMeasures,
    FeatureSetVisualization, NestedCVResults, NestedCrossValidation, PowerAnalysis,
    QualityAssessment, RedundancyMeasures, RelevanceScoring, StabilityMeasures, StratifiedKFold,
};

// pub use crate::group_selection::{
//     GraphStructuredSparsitySelector, GroupLassoSelector, HierarchicalStructuredSparsitySelector,
//     OverlappingGroupSparsitySelector, SparseGroupLassoSelector,
// };

pub use crate::statistical_tests::{
    chi2,
    f_classif,
    f_oneway,
    f_regression,
    kruskal_wallis,
    mann_whitney_u,
    mutual_info_classif,
    mutual_info_regression,
    // permutation_tests, spearman_correlation, transfer_entropy - commented out
    r_regression,
};

// pub use crate::streaming::{
//     ConceptDriftAwareSelector, OnlineFeatureSelector, StreamingFeatureImportance,
// };

// pub use crate::hierarchical::{
//     FeatureHierarchy, HierarchicalFeatureSelector, HierarchicalSelectionStrategy, HierarchyNode,
//     MultiLevelHierarchicalSelector, ScoreAggregation,
// };

pub use crate::multi_label::{
    AggregateMethod, LabelSpecificSelector, MultiLabelFeatureSelector, MultiLabelStrategy,
    MultiLabelTarget,
};

pub use crate::bayesian::{
    BayesianInferenceMethod, BayesianModelAveraging, BayesianVariableSelector, PriorType,
};

pub use crate::spectral::{
    GraphConstructionMethod, KernelFeatureSelector, KernelType, LaplacianScoreSelector,
    ManifoldFeatureSelector, ManifoldMethod, SpectralFeatureSelector,
};

pub use crate::optimization::{
    ADMMFeatureSelector, ConvexFeatureSelector, IntegerProgrammingFeatureSelector,
    ProximalGradientSelector, SemidefiniteFeatureSelector,
};

// TODO: ndarray 0.17 HRTB trait bound issues
// pub use crate::validation::{
//     DistributionalPropertyTest, PermutationSignificanceTest, RobustnessTest,
//     SelectionConsistencyTest, StatisticalValidationFramework, StatisticalValidationResults,
// };

pub use crate::parallel::{
    ParallelCorrelationComputer, ParallelFeatureEvaluator, ParallelFeatureRanker,
    ParallelSelectionUtils, ParallelUnivariateRegressionScorer, ParallelUnivariateScorer,
    ParallelVarianceComputer,
};

pub use crate::plugin::{
    ComputationalComplexity, FeatureSelectionPlugin, LoggingMiddleware, MemoryComplexity,
    PerformanceMetrics, PerformanceMiddleware, PipelineResult, PluginContext, PluginMetadata,
    PluginPipeline, PluginRegistry, PluginResult, StepResult as PluginStepResult,
};

pub use crate::pipeline::{
    BinningStrategy, FeatureSelectionPipeline, OptimizationConfiguration, PipelineConfiguration,
    PreprocessingStep, SelectionMethod, Trained, Untrained,
};

pub use crate::type_safe::{data_states, selection_types, FeatureIndex, FeatureMask};

pub use crate::performance::SIMDStats;

pub use crate::fluent_api::{
    presets, FeatureSelectionBuilder, FluentConfig, FluentSelectionResult, SelectionStep,
    StepResult,
};

pub use crate::comprehensive_benchmark::{
    quick_benchmark, BenchmarkConfiguration, BenchmarkDataset as ComprehensiveBenchmarkDataset,
    BenchmarkMethod, BenchmarkMetric, ComprehensiveBenchmarkResults, ComprehensiveBenchmarkSuite,
    DatasetDomain, DatasetMetadata, DetailedMethodResult, MethodCategory, TaskType,
};

#[cfg(feature = "serde")]
pub use crate::serialization::{exports, ExportFormat, SelectionResultsIO};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod domain_specific_tests;
