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
#![allow(ambiguous_glob_reexports)]
//! Dummy estimators for baseline comparisons
//!
//! This module provides simple baseline estimators that ignore the input features
//! and generate predictions based on simple rules. These are useful for establishing
//! baselines for comparison with more sophisticated models.
//!
//! The module includes:
//! - [`DummyClassifier`] - Simple rules-based classifier
//! - [`DummyRegressor`] - Simple rules-based regressor
//! - [`ContextAwareDummyRegressor`] - Context-aware baselines using feature information
//! - [`ContextAwareDummyClassifier`] - Context-aware classifier baselines
//! - [`RobustDummyRegressor`] - Robust baselines resistant to outliers
//! - [`RobustDummyClassifier`] - Robust classifier baselines
//! - [`OnlineDummyRegressor`] - Online learning regressor for streaming data
//! - [`OnlineDummyClassifier`] - Online learning classifier for streaming data
//! - [`BenchmarkClassifier`] - Standard benchmark baselines for classification
//! - [`BenchmarkRegressor`] - Standard benchmark baselines for regression

// #![warn(missing_docs)]

pub mod advanced_bayesian;
pub mod benchmark;
pub mod causal_inference;
pub mod comparative_analysis;
#[allow(non_snake_case)]
#[cfg(test)]
pub mod comparison_tests;
pub mod context_aware;
pub mod domain_specific;
pub mod dummy_classifier;
pub mod dummy_multioutput_regressor;
pub mod dummy_regressor;
pub mod ensemble_dummy;
pub mod extensibility;
pub mod fairness_ethics;
pub mod fluent_api;
pub mod game_theoretic;
pub mod information_theoretic;
pub mod integration_utilities;
pub mod memory_management;
pub mod meta_learning;
pub mod modular_design;
pub mod online;
pub mod performance;
pub mod performance_enhancements;
pub mod robust;
pub mod scalability;
pub mod sklearn_benchmarks;
pub mod sklearn_comparison;
pub mod type_safe;
pub mod validation;

pub use advanced_bayesian::{
    AdvancedBayesianStrategy, EmpiricalBayesEstimator, HierarchicalBayesEstimator,
    MCMCBayesEstimator, VariationalBayesEstimator,
};
pub use benchmark::{
    BenchmarkClassifier, BenchmarkRegressor, BenchmarkStrategy, CompetitionBaseline,
    DomainBenchmarkClassifier, DomainStrategy as BenchmarkDomainStrategy, TheoreticalBound,
    TheoreticalBounds,
};
pub use causal_inference::{
    CausalDiscoveryBaseline, CausalDiscoveryStrategy, CounterfactualBaseline,
    CounterfactualStrategy, DoCalculusBaseline, DoCalculusStrategy, FittedCausalDiscoveryBaseline,
    FittedCounterfactualBaseline, FittedDoCalculusBaseline, FittedInstrumentalVariableBaseline,
    FittedMediationAnalysisBaseline, InstrumentalVariableBaseline, InstrumentalVariableStrategy,
    MediationAnalysisBaseline, MediationStrategy,
};
pub use comparative_analysis::{
    ComparativeAnalyzer, ComparisonReporter, ConfidenceIntervalType, EffectSizeInterpretation,
    EffectSizeMeasure, EffectSizeResult, ModelComparisonResult, MultipleComparisonCorrection,
    PairwiseComparison, SignificanceTest, SignificanceTestResult, StatisticalSummary,
};
pub use context_aware::{
    ContextAwareDummyClassifier, ContextAwareDummyRegressor, ContextAwareStrategy, FeatureWeighting,
};
pub use domain_specific::{
    AnomalyFeatures, AnomalyStrategy, CVFeatures, CVStrategy, ColorSpace, DomainClassifier,
    DomainFeatures, DomainPreprocessor, DomainStrategy, FrequencyMethod, NLPFeatures, NLPStrategy,
    PixelStatistic, RecFeatures, RecStrategy, TSFeatures, TextureMethod, ThresholdMethod,
    TimeSeriesStrategy,
};
pub use dummy_classifier::{DummyClassifier, Strategy as ClassifierStrategy};
pub use dummy_multioutput_regressor::{
    MultiOutputDummyRegressor, MultiOutputStrategy, SingleOutputStrategy,
};
pub use dummy_regressor::{
    CyclicalMethod, DecompositionMethod, DummyRegressor, PredictConfidenceInterval,
    ProbabilisticRegression, SeasonalAdjustmentMethod, SeasonalType, Strategy as RegressorStrategy,
};
pub use ensemble_dummy::{EnsembleDummyClassifier, EnsembleDummyRegressor, EnsembleStrategy};
pub use extensibility::{
    BaselinePlugin, DataInfo, ErrorContext, ErrorHook, EvaluationFramework, EvaluationIntegration,
    EvaluationResult, FeatureType, FitContext, FitResult, HookSystem, LogLevel, LoggingConfig,
    MetricComputer, MetricResult, MetricType, MiddlewareContext, MiddlewareParameter,
    MiddlewarePipeline, MiddlewareResult, PipelineMiddleware, PluginConfig, PluginMetadata,
    PluginParameter, PluginRegistry, PostFitHook, PostPredictHook, PreFitHook, PrePredictHook,
    PredictContext, ResourceConfig, TargetType, TaskType, TestData,
};
pub use fairness_ethics::{
    BiasDetectionBaseline, BiasDetectionStrategy, BiasMetric, BiasMetricResult,
    DemographicParityBaseline, DemographicParityStrategy, EqualizedOddsBaseline,
    EqualizedOddsStrategy, FairnessAwareBaseline, FairnessConstraint, FairnessStrategy,
    FittedBiasDetectionBaseline, FittedDemographicParityBaseline, FittedEqualizedOddsBaseline,
    FittedFairnessAwareBaseline, FittedIndividualFairnessBaseline, GroupStatistics,
    IndividualFairnessBaseline, IndividualFairnessStrategy, SimilarityMetric, StatisticalTest,
    StatisticalTestResult,
};
pub use fluent_api::{
    ClassifierConfig, ClassifierFluentExt, ConfigPresets, PreprocessingChain, RegressorConfig,
    RegressorFluentExt,
};
pub use game_theoretic::{
    ExplorationStrategy, GameTheoreticClassifier, GameTheoreticRegressor, GameTheoreticResult,
    GameTheoreticStrategy, LpNorm, OpponentStrategy,
};
pub use information_theoretic::{
    EntropySamplingEstimator, InformationGainEstimator, InformationTheoreticStrategy, MDLEstimator,
    MaximumEntropyEstimator, MutualInformationEstimator,
};
pub use integration_utilities::{
    AutoBaselineGenerator, BaselineEstimator, BaselinePipeline, BaselineRecommendation,
    BaselineRecommendationEngine, BaselineType, ConfigurationHelper, DataCharacteristics,
    OptimizationHint, ParameterDefault, PerformanceMetrics as IntegrationPerformanceMetrics,
    PipelineConfig, PreprocessingStep, RecommendationRule, SmartDefaultSelector,
    ValidationStrategy,
};
pub use memory_management::{advanced_pooling, reference_counting, streaming_algorithms};
pub use meta_learning::{
    ContinualLearningBaseline, ContinualStrategy, DomainAdaptationBaseline,
    DomainAdaptationStrategy, FewShotBaselineClassifier, FewShotBaselineRegressor, FewShotStrategy,
    FittedContinualLearningBaseline, FittedDomainAdaptationBaseline, FittedFewShotClassifier,
    FittedFewShotRegressor, FittedTransferBaseline, SourceDomainStats, TransferLearningBaseline,
    TransferStrategy,
};
pub use modular_design::{
    statistical_methods, BaselineStrategy, BaselineStrategyFactory, ClassificationStrategy,
    ClippingPostprocessor, FittedPipeline, MeanConfig, MeanFittedData, MeanStrategy,
    MostFrequentConfig, MostFrequentFittedData, MostFrequentStrategy, Postprocessor,
    PredictionPipeline, Preprocessor, RegressionStrategy, StandardScaler, StrategyRegistry,
};
pub use online::{
    DriftDetectionMethod, OnlineClassificationStrategy, OnlineDummyClassifier,
    OnlineDummyRegressor, OnlineStrategy, WindowStrategy,
};
pub use performance::{benchmarks, cache_friendly, memory_efficient, parallel, simd_stats};
pub use performance_enhancements::{branch_optimization, cpu_optimization, dummy_optimization};
pub use robust::{
    LocationEstimator, OutlierDetectionMethod, RobustDummyClassifier, RobustDummyRegressor,
    RobustStrategy, ScaleEstimator,
};
pub use scalability::{
    ApproximateBaseline, ApproximateMethod, ApproximateStats, LargeScaleConfig,
    LargeScaleDummyEstimator, LargeScaleStrategy, ProcessingStats, SampledBaselineResult,
    SamplingBasedBaseline, StreamingBaselineUpdater,
};
pub use sklearn_benchmarks::{
    AccuracyComparison, BenchmarkConfig, BenchmarkResult, DatasetConfig, DatasetInfo,
    DatasetProperties, DatasetSize, DatasetType, NumericalAccuracy, PerformanceMetrics,
    SklearnBenchmarkFramework, TargetStatistics,
};
pub use sklearn_comparison::{
    generate_comparison_report, ComparisonResult, SklearnComparisonFramework,
};
pub use type_safe::{
    BoundedParameter, Classification, ClassificationFittedData, EstimatorConfig, EstimatorState,
    ParameterValidation, PositiveInt, Probability, RandomSeed, Regression, RegressionFittedData,
    StrategyValid, TaskType as TypeSafeTaskType, Trained, TypeSafeDummyEstimator,
    TypeSafeEstimator, TypeSafeFittedClassifier, TypeSafeFittedRegressor, TypeSafeParameters,
    Untrained, ValidatedStrategy,
};
pub use validation::{
    analyze_classification_dataset, analyze_regression_dataset, bootstrap_validate_classifier,
    bootstrap_validate_regressor, compare_dummy_strategies, comprehensive_validation_classifier,
    cross_validate_dummy, get_adaptive_classification_strategy, get_adaptive_regression_strategy,
    get_best_strategy, get_ranking_summary, get_strategies_in_tier, permutation_test_classifier,
    permutation_test_vs_random_classifier, rank_dummy_strategies_classifier,
    rank_dummy_strategies_regressor, recommend_classification_strategy,
    recommend_regression_strategy, validate_reproducibility, BootstrapValidationResult,
    ClassDistribution, DataType, DatasetCharacteristics, DummyValidationResult,
    PermutationTestResult, StatisticalValidationResult, StrategyRanking, StrategyRecommendation,
    TargetDistribution, ValidationSummary,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};
    use sklears_core::traits::{Fit, Predict};

    #[test]
    fn test_dummy_classifier_integration() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);
        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_dummy_regressor_integration() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let regressor = DummyRegressor::new(RegressorStrategy::Mean);
        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }
}
