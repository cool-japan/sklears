#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
//! Model selection utilities for sklears

mod adaptive_resource_allocation;
mod adversarial_validation;
mod automl_algorithm_selection;
mod automl_feature_engineering;
mod automl_pipeline;
mod bandit_optimization;
mod bayes_search;
mod bayesian_model_averaging;
mod bayesian_model_selection;
mod bias_variance;
mod config_management;
mod conformal_prediction;
mod cross_validation;
mod cv;
mod cv_model_selection;
mod drift_detection;
mod early_stopping;
mod ensemble_evaluation;
mod ensemble_selection;
mod epistemic_uncertainty;
mod evolutionary;
mod grid_search;
mod halving_grid_search;
mod hierarchical_validation;
mod hyperparameter_importance;
mod imbalanced_validation;
mod incremental_evaluation;
mod information_criteria;
mod memory_efficient;
mod meta_learning;
mod meta_learning_advanced;
mod model_comparison;
mod model_complexity;
mod multi_fidelity_advanced;
mod multi_fidelity_optimization;
mod multilabel_validation;
mod neural_architecture_search;
mod noise_injection;
mod ood_validation;
mod optimizer_plugins;
mod parallel_optimization;
mod parameter_space;
mod population_based_training;
mod scoring;
mod spatial_validation;
mod temporal_validation;
mod threshold_tuning;
mod train_test_split;
mod validation;
mod warm_start;
mod worst_case_validation;

pub use adaptive_resource_allocation::{
    AdaptiveAllocationConfig, AdaptiveResourceAllocator, AllocationStatistics, AllocationStrategy,
    ResourceConfiguration,
};
pub use adversarial_validation::{
    AdversarialStatistics, AdversarialValidationConfig, AdversarialValidationResult,
    AdversarialValidator,
};
pub use automl_algorithm_selection::{
    select_best_algorithm, AlgorithmFamily, AlgorithmSelectionResult, AlgorithmSpec,
    AutoMLAlgorithmSelector, AutoMLConfig, ComputationalConstraints,
    DatasetCharacteristics as AutoMLDatasetCharacteristics, RankedAlgorithm, TargetStatistics,
};
pub use automl_feature_engineering::{
    engineer_features, AutoFeatureEngineer, AutoFeatureEngineering, FeatureEngineeringResult,
    FeatureEngineeringStrategy, FeatureSelectionMethod, FeatureStatistics,
    FeatureTransformationType, GeneratedFeature, TransformationInfo,
};
pub use automl_pipeline::{
    automl, automl_with_budget, AutoMLPipeline, AutoMLPipelineConfig, AutoMLPipelineResult,
    AutoMLProgressCallback, AutoMLStage, ConsoleProgressCallback, OptimizationLevel,
};
pub use bandit_optimization::{
    BanditConfig, BanditOptimization, BanditOptimizationResult, BanditSearchCV, BanditStrategy,
};
pub use bayes_search::{
    AcquisitionFunction as BayesAcquisitionFunction, BayesSearchCV, BayesSearchConfig,
    ParamDistribution as BayesParamDistribution, TPEConfig, TPEOptimizer,
};
pub use bayesian_model_averaging::{
    bayesian_model_average, BMAConfig, BMAResult, BayesianModelAverager as BMA_Averager,
    EvidenceMethod, ModelInfo as BMA_ModelInfo, PriorType,
};
pub use bayesian_model_selection::{
    BayesianModelAverager as BayesianModelSelector_BA, BayesianModelSelectionResult,
    BayesianModelSelector, EvidenceEstimationMethod, ModelEvidenceData,
};
pub use bias_variance::{
    bias_variance_decompose, BiasVarianceAnalyzer, BiasVarianceConfig, BiasVarianceResult,
    SampleBiasVariance,
};
pub use config_management::{
    ConfigError, ConfigManager, CrossValidationConfig as CVConfig, EarlyStoppingConfig as ESConfig,
    ModelSelectionConfig, OptimizationConfig as OptiConfig, ParameterDefinition, ResourceConfig,
    ScoringConfig as ScoreConfig,
};
pub use conformal_prediction::{
    ConformalPredictionConfig, ConformalPredictionResult, ConformalPredictor, CoverageStatistics,
    EfficiencyMetrics, JackknifeConformalPredictor, NonconformityMethod,
};
pub use cross_validation::{
    BlockCrossValidator, BlockedTimeSeriesCV, BootstrapCV, CrossValidator, CustomCrossValidator,
    GroupKFold, GroupShuffleSplit, GroupStrategy, KFold, LeaveOneGroupOut, LeaveOneOut,
    LeavePGroupsOut, LeavePOut, MonteCarloCV, PredefinedSplit, PurgedGroupTimeSeriesSplit,
    RegressionCrossValidator, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit,
    StratifiedGroupKFold, StratifiedKFold, StratifiedRegressionKFold, StratifiedShuffleSplit,
    TimeSeriesSplit,
};
pub use cv_model_selection::{
    cv_select_model, CVModelScore, CVModelSelectionConfig, CVModelSelectionResult, CVModelSelector,
    ModelComparisonPair, ModelRanking, ModelSelectionCriteria,
};
pub use drift_detection::{
    DriftDetectionConfig, DriftDetectionMethod, DriftDetectionResult, DriftDetector,
    DriftStatistics,
};
pub use early_stopping::{
    AdaptationConfig, AdaptiveEarlyStopping, ConvergenceMetrics, EarlyStoppingCallback,
    EarlyStoppingConfig, EarlyStoppingMonitor, EarlyStoppingStrategy,
};
pub use ensemble_evaluation::{
    evaluate_ensemble, DiversityAnalysis, DiversityMeasure, EnsembleEvaluationConfig,
    EnsembleEvaluationResult, EnsembleEvaluationStrategy, EnsembleEvaluator,
    EnsemblePerformanceMetrics, MemberContribution, MultiObjectiveAnalysis, OutOfBagScores,
    ProgressivePerformance, StabilityAnalysis, StabilityMetric,
};
pub use ensemble_selection::{
    select_ensemble, DiversityMeasures, EnsemblePerformance, EnsembleSelectionConfig,
    EnsembleSelectionResult, EnsembleSelector, EnsembleStrategy, ModelInfo as EnsembleModelInfo,
    ModelPerformance,
};
pub use epistemic_uncertainty::{
    quantify_aleatoric_uncertainty, quantify_epistemic_uncertainty, quantify_uncertainty,
    AleatoricUncertaintyConfig, AleatoricUncertaintyMethod, AleatoricUncertaintyQuantifier,
    AleatoricUncertaintyResult, CalibrationMethod, EpistemicUncertaintyConfig,
    EpistemicUncertaintyMethod, EpistemicUncertaintyQuantifier, EpistemicUncertaintyResult,
    ReliabilityDiagram, ReliabilityMetrics, UncertaintyComponents, UncertaintyDecomposition,
    UncertaintyDecompositionMethod, UncertaintyQuantificationConfig,
    UncertaintyQuantificationResult, UncertaintyQuantifier,
};
pub use evolutionary::{
    EvolutionarySearchCV, EvolutionarySearchResult, GeneticAlgorithmCV, GeneticAlgorithmConfig,
    GeneticAlgorithmResult, Individual, MultiObjectiveGA, MultiObjectiveResult, ParameterDef,
    ParameterSpace as EvolutionaryParameterSpace,
};
pub use grid_search::{
    GridSearchCV, GridSearchResults, ParameterDistribution, ParameterDistributions, ParameterGrid,
    ParameterSet, ParameterValue as GridParameterValue, RandomizedSearchCV,
};
pub use halving_grid_search::{
    HalvingGridSearch, HalvingGridSearchConfig, HalvingGridSearchResults, HalvingRandomSearchCV,
};
pub use hierarchical_validation::{
    hierarchical_cross_validate, ClusterInfo, HierarchicalCrossValidator, HierarchicalSplit,
    HierarchicalStrategy, HierarchicalValidationConfig, HierarchicalValidationResult,
};
pub use hyperparameter_importance::{
    analyze_parameter_sensitivity, compute_shap_importance, AblationAnalyzer, AblationConfig,
    AblationResult, ComprehensiveImportanceResult, FANOVAAnalyzer, FANOVAConfig, FANOVAResult,
    HyperparameterImportanceAnalyzer, ParameterSensitivity, SHAPAnalyzer, SHAPConfig, SHAPResult,
    SensitivityAnalyzer, SensitivityConfig, SensitivityResult,
};
pub use imbalanced_validation::{
    imbalanced_cross_validate, ClassStatistics, ImbalancedCrossValidator, ImbalancedSplit,
    ImbalancedStrategy, ImbalancedValidationConfig, ImbalancedValidationResult, SamplingStrategy,
};
pub use incremental_evaluation::{
    evaluate_incremental_stream, AdaptationCriterion, ConceptDriftHandling, DriftDetectorType,
    DriftEvent, DriftType, FoldUpdateStrategy, IncrementalEvaluationConfig,
    IncrementalEvaluationResult, IncrementalEvaluationStrategy, IncrementalEvaluator,
    PerformanceSnapshot, StreamingStatistics,
};
pub use information_criteria::{
    CrossValidatedIC, InformationCriterion, InformationCriterionCalculator,
    InformationCriterionResult, ModelComparisonResult as ICModelComparisonResult,
};
pub use memory_efficient::{
    memory_efficient_cross_validate, DataChunk, MemoryEfficiencyStats, MemoryEfficientConfig,
    MemoryEfficientCrossValidator, MemoryError, MemoryPool, MemorySnapshot, MemoryTracker,
    StreamingDataReader, StreamingEvaluationResult,
};
pub use meta_learning::{
    meta_learning_recommend, ComplexityMeasures, DatasetCharacteristics, FeatureType,
    MetaLearningConfig, MetaLearningEngine, MetaLearningRecommendation, MetaLearningStrategy,
    OptimizationRecord, ParameterValue, SimilarityMetric, StatisticalMeasures, SurrogateModel,
    TransferMethod,
};
pub use meta_learning_advanced::{
    Experience as OptimizationExperience_Advanced, ExperienceReplayBuffer, ExperienceReplayConfig,
    FewShotAlgorithm, FewShotConfig, FewShotOptimizer, FewShotResult, ImportanceWeightingMethod,
    Learn2OptimizeConfig, Learn2OptimizeResult, LearnedOptimizer,
    OptimizationExperience as MetaOptimizationExperience, OptimizationLearner,
    OptimizationTask as MetaOptimizationTask, OptimizerArchitecture,
    ParameterRange as MetaParameterRange, ParameterScale, PrioritizationStrategy, ReplayResult,
    SamplingStrategy as ReplaySamplingStrategy, TaskCharacteristics as MetaTaskCharacteristics,
    TransferLearningConfig, TransferLearningOptimizer, TransferResult, TransferStrategy,
};
pub use model_comparison::{
    friedman_test, mcnemar_test, multiple_model_comparison, nemenyi_post_hoc_test, paired_t_test,
    wilcoxon_signed_rank_test, ModelComparisonResult, MultipleTestingCorrection,
    StatisticalTestResult,
};
pub use model_complexity::{
    analyze_model_complexity, detect_overfitting_learning_curve, ComplexityAnalysisConfig,
    ComplexityAnalysisResult, ComplexityMeasure, ComplexityRecommendation, ModelComplexityAnalyzer,
    OverfittingDetector,
};
pub use multi_fidelity_advanced::{
    AdaptiveFidelitySelector, AdaptiveFidelityStrategy, AllocationPlan, BudgetAllocationStrategy,
    BudgetAllocator, CoarseToFineConfig, CoarseToFineOptimizer, CoarseToFineResult,
    CoarseToFineStrategy, ConfigAllocation, ConfigurationWithPerformance,
    ProgressiveAllocationConfig, ProgressiveAllocationStrategy, ProgressiveAllocator,
};
pub use multi_fidelity_optimization::{
    multi_fidelity_optimize, AcquisitionFunction, CorrelationModel, CostModel, FidelityEvaluation,
    FidelityLevel, FidelityProgression, FidelitySelectionMethod, MultiFidelityConfig,
    MultiFidelityOptimizer, MultiFidelityResult, MultiFidelityStrategy,
};
pub use multilabel_validation::{
    multilabel_cross_validate, LabelStatistics, MultiLabelCrossValidator, MultiLabelSplit,
    MultiLabelStrategy, MultiLabelValidationConfig, MultiLabelValidationResult,
};
pub use neural_architecture_search::{
    ArchitectureEvaluation, ArchitectureSearchSpace, NASConfig, NASOptimizer, NASResult,
    NASStrategy, NeuralArchitecture,
};
pub use noise_injection::{
    robustness_test, AdversarialMethod, NoiseConfig, NoiseInjector, NoiseStatistics, NoiseType,
    RobustnessTestResult,
};
pub use ood_validation::{
    validate_ood, DistributionShiftMetrics, OODConfidenceIntervals, OODDetectionMethod,
    OODValidationConfig, OODValidationResult, OODValidator,
};
pub use optimizer_plugins::{
    CustomMetric, Evaluation, HookError, HookManager, LoggingHook, MetricError, MetricRegistry,
    MiddlewareError, MiddlewarePipeline, NormalizationMiddleware,
    OptimizationHistory as PluginOptimizationHistory, OptimizationHook, OptimizationMiddleware,
    OptimizerPlugin, ParameterConstraints as PluginParameterConstraints, PluginConfig, PluginError,
    PluginFactory, PluginRegistry, StopReason,
};
pub use parallel_optimization::{
    parallel_optimize, BatchAcquisitionStrategy, CommunicationProtocol, ErrorHandlingStrategy,
    EvaluationResult, LoadBalancingStrategy, ParallelOptimizationConfig,
    ParallelOptimizationResult, ParallelOptimizer, ParallelStrategy, ProgressReportingConfig,
    ResourceUtilization, SynchronizationStrategy, WorkerStatistics,
};
pub use parameter_space::{
    CategoricalParameter, ConditionalParameter, ParameterConstraint, ParameterImportanceAnalyzer,
    ParameterSpace,
};
pub use population_based_training::{
    PBTConfig, PBTConfigFn, PBTParameterSpace, PBTParameters, PBTResult, PBTStatistics, PBTWorker,
    PopulationBasedTraining, PopulationBasedTrainingCV,
};
pub use scoring::{
    paired_ttest, ClosureScorer, CustomScorer, EnhancedScorer, ScorerRegistry, ScoringConfig,
    ScoringResult, SignificanceTestResult, TaskType,
};
pub use spatial_validation::{
    DistanceMethod, LeaveOneRegionOut, SpatialClusteringMethod, SpatialCoordinate,
    SpatialCrossValidator, SpatialValidationConfig,
};
pub use temporal_validation::{
    BlockedTemporalCV, SeasonalCrossValidator, TemporalCrossValidator, TemporalValidationConfig,
};
pub use threshold_tuning::{
    optimize_threshold, FixedThresholdClassifier, OptimizationMetric, ThresholdOptimizationResult,
    TunedThresholdClassifierCV, TunedThresholdClassifierCVTrained,
};
pub use train_test_split::train_test_split;
pub use validation::{
    cross_val_predict, cross_val_score, cross_validate, learning_curve, nested_cross_validate,
    permutation_test_score, validation_curve, CrossValidateResult, LearningCurveResult,
    NestedCVResult, ParamConfigFn, PermutationTestResult, Scoring, ValidationCurveResult,
};
pub use warm_start::{
    EvaluationRecord, OptimizationHistory, OptimizationStatistics, TransferLearning,
    WarmStartConfig, WarmStartInitializer, WarmStartStrategy,
};
pub use worst_case_validation::{
    worst_case_validate, AdversarialAttackMethod, CorruptionType, DistributionShiftType,
    DriftPattern, MissingPattern, NoisePattern, RobustnessMetrics, ScenarioResult,
    WorstCaseScenario, WorstCaseScenarioGenerator, WorstCaseValidationConfig,
    WorstCaseValidationResult, WorstCaseValidator,
};
