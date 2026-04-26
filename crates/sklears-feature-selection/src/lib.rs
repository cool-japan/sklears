//! Feature selection algorithms
//!
//! This module provides algorithms for selecting relevant features from data,
//! compatible with scikit-learn's feature_selection module.
//!
//! All major modules are now enabled, including:
//! - `comparison_tests` - Comparison tests against reference implementations
//! - `embedded` - Embedded feature selection methods (L1-based, tree-based, stability selection)
//! - `ensemble_selectors` - Ensemble-based feature selection (Boruta, Bootstrap)
//! - `genetic_optimization` - Genetic algorithm-based multi-objective feature selection
//! - `tree_based_selectors` - Tree importance-based feature selectors
//! - `validation` - Statistical validation framework for feature selection
//! - `wrapper` - Wrapper methods (RFE, RFECV, SelectFromModel)
//!
//! ## Technical Note: ndarray 0.17 Fix
//!
//! ndarray 0.17 added a 3rd default type parameter `A = <S as RawData>::Elem` to `ArrayBase`.
//! The Rust trait solver previously could not normalize `Array2<Float>` (2-param alias) to
//! `ArrayBase<OwnedRepr<f64>, Dim<[usize;2]>, f64>` (3-param form) in where bounds with
//! any remaining generic type parameters. Fixed by using fully expanded 3-parameter form
//! in all where bounds via type aliases (Mat2f64, Vec1f64, etc.) and macro-based impls.

pub mod automl;
pub mod base;
pub mod bayesian;
pub mod benchmark;
pub mod comparison_tests;
pub mod comprehensive_benchmark;
pub mod domain_benchmark;
pub mod domain_specific;
pub mod embedded;
pub mod ensemble_selectors;
pub mod evaluation;
pub mod filter;
pub mod fluent_api;
pub mod genetic_optimization;
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
pub mod tree_based_selectors;
pub mod type_safe;
pub mod validation;
pub mod wrapper;

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

pub use crate::wrapper::{
    CrossValidator, FeatureImportance, HasCoefficients, IndexableTarget, KFold, RFECVResults,
    SelectFromModel, SequentialFeatureSelector, RFE, RFECV,
};

pub use crate::embedded::{
    ConsensusFeatureSelector, ConsensusMethod, ConsensusThresholdParams, MultiTaskFeatureSelector,
    StabilitySelector,
};

pub use crate::genetic_optimization::{
    CostSensitiveObjective, FairnessAwareObjective, FairnessMetric, FeatureCountObjective,
    FeatureDiversityObjective, FeatureImportanceObjective, GeneticSelector, Individual,
    MultiObjectiveFeatureSelector, MultiObjectiveMethod, ObjectiveFunction,
    PredictivePerformanceObjective,
};

pub use crate::ensemble_selectors::{
    extract_features, AggregationMethod, BootstrapSelector, BorutaSelector, EnsembleFeatureRanking,
    SelectorFunction, UnivariateMethod, UnivariateSelector,
};

pub use crate::tree_based_selectors::{
    GradientBoostingSelector, TreeImportance, TreeImportanceSelector, TreeSelector,
};

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

pub use crate::validation::{
    DistributionalPropertyTest, PermutationSignificanceTest, RobustnessTest,
    SelectionConsistencyTest, StatisticalValidationFramework, StatisticalValidationResults,
};

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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

#[cfg(test)]
mod domain_specific_tests;
