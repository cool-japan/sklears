//! Evaluation metrics for feature selection quality assessment
//!
//! This module provides comprehensive evaluation and validation capabilities for feature selection
//! methods including stability measures, consistency metrics, redundancy assessment, quality
//! indicators, cross-validation strategies, and advanced statistical analysis. All algorithms
//! have been refactored into focused modules for better maintainability and comply with SciRS2 Policy.

// Stability measures for feature selection consistency
mod stability_measures;
pub use stability_measures::{
    ConsistencyIndex, DiceSimilarity, JaccardSimilarity, OverlapCoefficient, StabilityMeasures,
};

// Redundancy assessment for feature set quality
mod redundancy_measures;
pub use redundancy_measures::{
    CorrelationRedundancy, MutualInformationRedundancy, RedundancyAssessment, RedundancyMatrix,
    RedundancyMeasures, VarianceInflationFactor,
};

// Relevance scoring methods
mod relevance_scoring;
pub use relevance_scoring::{
    ChiSquareScoring, CorrelationScoring, FStatisticScoring, InformationGainScoring,
    RelevanceAssessment, RelevanceScoring, ReliefScoring,
};

// Quality assessment metrics
mod quality_assessment;
pub use quality_assessment::{
    ComprehensiveQualityAssessment, InterpretabilityMetrics, ModelComplexity,
    PredictivePerformance, QualityAssessment, QualityAssessmentResult, SelectionQuality,
};

// Feature set visualization utilities
mod feature_visualization;
pub use feature_visualization::{
    FeatureImportancePlots, FeatureSetVisualization, RedundancyHeatmaps, SelectionFrequencyCharts,
    StabilityPlots,
};

// Cross-validation strategies
mod cross_validation;
pub use cross_validation::{
    FeatureStabilityMetrics, GroupKFold, InnerCVResult, NestedCVResults, NestedCrossValidation,
    RepeatedKFold, StratifiedKFold, TimeSeriesSplit,
};

// Feature set diversity measures
mod diversity_measures;
pub use diversity_measures::{
    DiversityIndex, DiversityMatrix, EnsembleDiversity, FeatureSetDiversityMeasures, FeatureSpacing,
};

// Feature interaction analysis
mod interaction_analysis;
pub use interaction_analysis::{
    FeatureInteractionAnalysis, HigherOrderInteractions, InteractionStrength, PairwiseInteractions,
    SynergyDetection,
};

// Statistical significance testing
mod statistical_testing;
pub use statistical_testing::{
    MultipleComparisonsCorrection, PermutationTests, PowerAnalysis, SignificanceAnalysis,
    StatisticalTesting,
};

// Benchmark and comparison utilities
mod benchmarking;
pub use benchmarking::{
    BenchmarkSuite, ComparativeAnalysis, FeatureSelectionBenchmark, MethodComparison,
    PerformanceRanking,
};

// Robustness evaluation
mod robustness_evaluation;
pub use robustness_evaluation::{
    NoiseResistance, OutlierSensitivity, ParameterSensitivity, RobustnessEvaluation,
    StabilityUnderPerturbation,
};

// Evaluation metrics aggregation
mod metrics_aggregation;
pub use metrics_aggregation::{
    ConsensusMetrics, MetricsAggregator, MultiCriteriaEvaluation, RankAggregation,
    WeightedAveraging,
};

// Advanced evaluation techniques
mod advanced_evaluation;
pub use advanced_evaluation::{
    BayesianEvaluation, CausalFeatureEvaluation, DomainSpecificEvaluation, OnlineEvaluation,
    TransferLearningEvaluation,
};
