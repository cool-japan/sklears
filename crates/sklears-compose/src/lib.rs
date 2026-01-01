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
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
//! Composite estimators and transformers
//!
//! This module provides meta-estimators for composing other estimators.
//! It includes tools for applying different transformers to different
//! subsets of features and for transforming target variables.

// #![warn(missing_docs)]

// Re-export core modules
pub mod advanced_debugging;
pub mod advanced_pipeline;
pub mod api_consistency;
pub mod automated_alerting;
pub mod automl;
pub mod benchmarking;
pub mod boosting;
pub mod circuit_breaker;
pub mod column_transformer;
pub mod error;
// pub mod composable_execution;  // Migrated to execution module
pub mod config_management;
pub mod configuration_validation;
pub mod continual_learning;
// TODO: Temporarily disabled until ndarray 0.17 HRTB trait bound issues are resolved
// pub mod cross_validation;
pub mod cv_pipelines;
pub mod dag_pipeline;
pub mod debugging;
pub mod debugging_utilities;
pub mod developer_experience;
pub mod differentiable;
pub mod distributed;
pub mod distributed_tracing;
pub mod enhanced_compile_time_validation;
pub mod enhanced_error_messages;
pub mod enhanced_errors;
pub mod ensemble;
pub mod execution;
pub mod execution_config;
pub mod execution_core;
pub mod execution_hooks;
pub mod execution_strategies;
pub mod execution_types;
pub mod external_integration;
pub mod fault_core;
pub mod feature_engineering;
pub mod few_shot;
pub mod fluent_api;
pub mod memory_optimization;
pub mod meta_learning;
pub mod middleware;
pub mod mock;
pub mod modular_framework;
pub mod monitoring;
pub mod nlp_pipelines;
pub mod optimization;
pub mod parallel_execution;
// pub mod pattern_optimization; // Temporarily disabled - needs rand_distr dependency and type fixes
pub mod performance_optimization;
pub mod performance_profiler;
pub mod performance_testing;
pub mod pipeline;
pub mod pipeline_visualization;
pub mod plugin_architecture;
pub mod profile_guided_optimization;
pub mod property_testing;
pub mod quality_assurance;
pub mod quantum;
pub mod resource_management;
// pub mod retry; // Temporarily disabled - needs missing type imports and context fixes
pub mod enhanced_wasm_integration;
pub mod scheduling;
pub mod simd_optimizations;
pub mod state_management;
pub mod streaming;
pub mod stress_testing;
pub mod task_definitions;
pub mod task_scheduling;
pub mod time_series_pipelines;
pub mod transfer_learning;
pub mod type_safety;
pub mod validation;
pub mod wasm_integration;
#[cfg(feature = "workflow")]
pub mod workflow_language;
pub mod zero_cost;

// Re-export main types and traits
pub use advanced_debugging::{
    AdvancedPipelineDebugger, AdvancedProfiler, Breakpoint, BreakpointCondition, CallStackFrame,
    CpuSample, DebugConfig, DebugEvent, DebugSession, DebugSessionHandle, DebugSessionState,
    DebugStatistics, ExecutionStep, MemorySample, StepResult, VariableInspector, VariableValue,
    WatchExpression, WatchResult,
};
pub use advanced_pipeline::{
    BranchConfig, BranchingPipeline, BranchingPipelineBuilder, BranchingPipelineTrained,
    ConditionalPipeline, ConditionalPipelineBuilder, ConditionalPipelineTrained, DataCondition,
    FeatureCountCondition,
};
pub use api_consistency::{
    ApiConsistencyChecker, ApiRecommendation, ConfigSummary, ConfigValue, ConsistencyIssue,
    ConsistencyReport, ExecutionMetadata, FittedModelSummary, FittedTransformerSummary,
    IssueCategory, IssueSeverity, MetadataProvider, ModelSummary, PipelineConsistencyReport,
    RecommendationCategory, RecommendationPriority, StandardBuilder, StandardConfig,
    StandardEstimator, StandardFittedEstimator, StandardFittedTransformer, StandardResult,
    StandardTransformer,
};
pub use automated_alerting::{
    ActiveAlert, AlertChannel, AlertCondition, AlertConfig, AlertEvent, AlertSeverity, AlertStatus,
    AutomatedAlerter, ConsoleAlertChannel, EmailAlertChannel, EscalationLevel, LogicalOperator,
    PatternField, SilencePeriod, SlackAlertChannel, ThresholdOperator, WebhookAlertChannel,
};
pub use automl::{
    ActivationFunction, AlgorithmChoice, AlgorithmType, AutoMLConfig, AutoMLOptimizer, LayerType,
    NASStrategy, NeuralArchitecture, NeuralArchitectureSearch, NeuralSearchSpace,
    OptimizationHistory, OptimizationMetric, OptimizationReport, ParameterRange,
    ParameterValue as AutoMLParameterValue, SearchSpace, SearchStrategy as AutoMLSearchStrategy,
    TrialResult, TrialStatus,
};
pub use benchmarking::{
    BenchmarkConfig, BenchmarkReport, BenchmarkResult, BenchmarkSuite, ComplexityClass,
    MemoryUsage as BenchmarkMemoryUsage, ScalabilityMetrics, UseCase,
};
pub use boosting::{
    AdaBoostAlgorithm, AdaBoostClassifier, AdaBoostTrained, GradientBoostingRegressor,
    GradientBoostingTrained, LossFunction,
};
pub use circuit_breaker::{
    AdvancedCircuitBreaker, AnalyticsInsight, AnalyticsProcessor, AnalyticsRecommendation,
    AnalyticsResult, CircuitBreaker, CircuitBreakerAnalytics, CircuitBreakerBuilder,
    CircuitBreakerError, CircuitBreakerEvent, CircuitBreakerEventRecorder, CircuitBreakerEventType,
    CircuitBreakerFailureDetector, CircuitBreakerRecoveryManager, CircuitBreakerStatsAggregator,
    CircuitBreakerStatsTracker, ConsoleEventPublisher, ErrorTracker, EventPublisher,
    FileEventPublisher, HealthCheckResult, HealthMetrics, RecoveryContext, RecoveryResult,
    RecoveryStrategy as CircuitBreakerRecoveryStrategy, RequestCounters, ResponseTimeTracker,
    SlidingWindow, ValidationResult as CircuitBreakerValidationResult,
};
pub use column_transformer::{
    ColumnTransformer, ColumnTransformerBuilder, ColumnTransformerOutput, ColumnTransformerTrained,
};
pub use config_management::{
    ConfigManager, ConfigValue as ConfigManagementConfigValue, EnvironmentConfig, EstimatorConfig,
    ExecutionConfig, PipelineConfig, ResourceConfig, StepConfig,
    ValidationRule as ConfigValidationRule,
};
pub use configuration_validation::{
    CompileTimeValidator, ConfigurationValidator, Constraint, CustomValidationRule,
    DependencyConstraint, FieldConstraints, FieldType, PipelineConfigValidator, RuleType,
    RuntimeValidator, ValidationBuilder, ValidationReport, ValidationResult, ValidationSchema,
    ValidationSeverity, ValidationStatus, ValidationSummary,
};
pub use continual_learning::{
    ContinualLearningPipeline, ContinualLearningPipelineTrained, ContinualLearningStrategy,
    MemoryBuffer, MemorySample as ContinualMemorySample, SamplingStrategy, Task, TaskStatistics,
};
// TODO: Temporarily disabled until ndarray 0.17 HRTB trait bound issues are resolved
// pub use cross_validation::{
//     CVStrategy, CVSummary, ComposedModelCrossValidator, CrossValidationConfig,
//     CrossValidationResults, FoldResult, NestedCVResults, OuterFoldResult, ScoringConfig,
//     ScoringMetric as CVScoringMetric, TimeSeriesConfig,
// };
pub use cv_pipelines::{
    AdaptationAlgorithm, AdaptationMetric, AdaptiveQualityConfig, BoundingBox,
    BufferManagementConfig, CVConfig, CVMetrics, CVModel, CVPipeline, CVPipelineState,
    CVPrediction, CacheEvictionPolicy, CachingStrategy, CameraInfo, CameraIntrinsics,
    CameraSettings, ColorSpace, CompressionAlgorithm, CompressionConfig, ComputeDevice,
    ConfidenceScores, ContrastiveLearningConfig, CrossModalLearningConfig, CrossModalStrategy,
    DenoisingAlgorithm, Detection, DetectionMetadata, DistillationConfig, EncodingConfig,
    ErrorResilienceConfig, ExifData, ExtractorConfig, ExtractorType, FeatureExtractor,
    FeatureMetadata, FeatureQuality, FeatureStatistics, FeatureVector,
    FusionStrategy as CVFusionStrategy, GPSInfo, ImageData, ImageDataType, ImageFormat,
    ImageMetadata, ImageSpecification, ImageTransform, InterpolationMethod, LensInfo,
    LoadBalancingAlgorithm, MemoryOptimizationLevel, Modality, ModelConfig, ModelMetadata,
    ModelPerformance, ModelType, MultiModalConfig, NetworkOptimizationConfig, NoiseReductionConfig,
    NormalizationSpec, ObjectDetectionResult, ParallelProcessingConfig,
    ParallelStrategy as CVParallelStrategy, PerformanceConfig, PerformanceMetrics,
    PostProcessor as CVPostProcessor, PredictionMetadata, PredictionResult, PredictionType,
    ProcessedResult, ProcessingComplexity, ProcessingMode, ProcessingStatistics, ProcessorConfig,
    ProcessorType, QualityEnhancementConfig, QualityLevel, QualityMetrics as CVQualityMetrics,
    QualitySettings, RateControlMethod, RealTimeProcessingConfig,
    RecoveryStrategy as CVRecoveryStrategy, ResourceUtilization, SharpeningConfig, StreamingConfig,
    StreamingProtocol, SyncMethod, TemporalAlignmentConfig, TransformParameter, VideoCodec,
};
pub use dag_pipeline::{
    BranchCondition, ComparisonOp, DAGNode, DAGPipeline, DAGPipelineTrained,
    ExecutionRecord as DAGExecutionRecord, ExecutionStats, MergeStrategy, NodeComponent,
    NodeConfig, NodeOutput,
};
pub use debugging::{
    Bottleneck as DebuggingBottleneck, BottleneckDetector,
    BottleneckSeverity as DebuggingBottleneckSeverity, BottleneckType,
    Breakpoint as DebuggingBreakpoint, BreakpointCondition as DebuggingBreakpointCondition,
    BreakpointFrequency, ComparisonOperator, DataSnapshot, DataSummary,
    DebugConfig as DebuggingConfig, DebugLogLevel, DebugOutputFormat,
    DebugSession as DebuggingSession, ErrorAnalysis, ErrorPattern, ErrorResolutionStatus,
    ErrorStatistics, ErrorTracker as DebuggingErrorTracker, ExecutionState,
    ExecutionStep as EnhancedExecutionStep, InteractiveDebugger, IoStatistics,
    MemoryUsage as DebuggingMemoryUsage, PerformanceAnalysis as DebuggingPerformanceAnalysis,
    PerformanceMeasurement, PerformanceMetric, PerformanceProfiler, PipelineDebugger, SessionState,
    StatisticalSummary, StepError, TrackedError,
};
// Temporarily commented out to avoid import conflicts
// pub use debugging_utilities::{
//     BottleneckAnalysis, BreakpointCondition as UtilsBreakpointCondition, CacheStatistics,
//     ContextValue, DataFlowAnalysis, DataInspector, DataLineageNode, DataMetadata,
//     DataSnapshot as UtilsDataSnapshot, DataStatistics, DebugCommand, DebugReport,
//     DebugSession as UtilsDebugSession, DebugState, DebuggingConfig, ErrorContext,
//     ErrorContextManager, ErrorInfo, ErrorSuggestion, ExecutionContext as UtilsExecutionContext,
//     ExecutionState as UtilsExecutionState, ExecutionStatistics,
//     ExecutionStep as UtilsExecutionStep, ExecutionTracer, ExportFormat, InspectionConfig,
//     InteractiveConfig, InteractiveDebugger as UtilsInteractiveDebugger, MeasurementSession,
//     PerformanceMetrics as UtilsPerformanceMetrics, PerformanceProfiler as UtilsPerformanceProfiler,
//     PipelineDebugger as UtilsPipelineDebugger, ProfilingConfig, QualityMetric, StepStatus,
//     StepType as UtilsStepType, TracingConfig, TransformationGraph, TransformationNode,
//     TransformationSummary, WatchExpression as UtilsWatchExpression,
// };
pub use developer_experience::{
    Breakpoint as DeveloperBreakpoint, CodeExample, DebugState, DebugSummary,
    DeveloperFriendlyError, ErrorMessageEnhancer, ExecutionContext, FixSuggestion,
    PipelineDebugger as DeveloperPipelineDebugger, StepType, SuggestionPriority, TraceEntry,
    WatchExpression as DeveloperWatchExpression,
};
pub use differentiable::{
    ActivationFunction as DiffActivationFunction, AutoDiffConfig, AutoDiffEngine, ComputationGraph,
    ComputationNode, DifferentiableOperation, DifferentiablePipeline, DifferentiableStage,
    DifferentiationMode, DualNumber, GradientAccumulation, GradientContext, GradientRecord,
    LearningRateSchedule as DiffLearningRateSchedule, NeuralPipelineController,
    OptimizationConfig as DiffOptimizationConfig, OptimizerState,
    OptimizerType as DiffOptimizerType, Parameter as DiffParameter,
    ParameterConfig as DiffParameterConfig, PipelineComponent, TrainingMetrics, TrainingState,
};
pub use distributed::TaskPriority as DistributedTaskPriority;
pub use distributed::{
    ClusterManager, ClusterNode, DataShard, DistributedTask, FaultDetector, LoadBalancer,
    MapReducePipeline, NodeStatus, ResourceRequirements, TaskResult as DistributedTaskResult,
    TaskStatus as DistributedTaskStatus,
};
pub use distributed_tracing::{
    Bottleneck, BottleneckSeverity as TracingBottleneckSeverity, ConsoleTraceExporter,
    DistributedTracer, JsonFileTraceExporter, LogEntry, LogLevel as TracingLogLevel,
    ServiceAnalysis, ServiceInfo, SpanStatus, Trace, TraceAnalysis, TraceExporter, TraceHandle,
    TraceSpan, TraceStatistics, TracingConfig,
};
pub use enhanced_compile_time_validation::{
    BuilderConfig, CompileTimeValidator as EnhancedCompileTimeValidator, ConfigType,
    ConfigValue as EnhancedConfigValue, ConfigurationLocation, ConstraintValidator,
    CrossReferenceRule, CrossReferenceValidator, CustomValidator, DependencyValidator,
    FieldDefinition, ParameterConstraintValidator, PipelineConfigurationSchema,
    PipelineSchemaValidator, ReferenceType, SchemaConstraint, SchemaConstraintType,
    SchemaValidator, SuggestionAction, SuggestionPriority as ValidationSuggestionPriority,
    TypeConstraint, TypeSafeConfigBuilder, Unbuilt, Unvalidated, Validated,
    ValidatedPipelineConfig, ValidationConfig, ValidationError, ValidationErrorCategory,
    ValidationMetrics, ValidationProof, ValidationResult as EnhancedValidationResult,
    ValidationRule, ValidationSeverity as EnhancedValidationSeverity,
    ValidationStatus as EnhancedValidationStatus, ValidationSuggestion, ValidationWarning,
    ValueConstraint, WarningCategory,
};
pub use enhanced_error_messages::{
    ActionableSuggestion, AutoRecoveryHandler, CodeExample as EnhancedCodeExample,
    ConfigurationContext, ContextProvider, ContextType, DataContext, DataQualityMetrics,
    DataStatistics, DifficultyLevel, DocumentationLink, EnhancedErrorContext, EnhancedErrorMessage,
    EnvironmentContext, ErrorCategory, ErrorClassification, ErrorContextCollector,
    ErrorEnhancementConfig, ErrorEnhancementStatistics, ErrorFormatter, ErrorFrequency,
    ErrorMessageEnhancer as EnhancedErrorMessageEnhancer, ErrorPattern as EnhancedErrorPattern,
    ErrorPatternAnalyzer, ExpertiseLevel, ImplementationStep, IssueRelationship, MissingValueInfo,
    OutputFormat, PerformanceBottleneck, PerformanceContext as EnhancedPerformanceContext,
    PipelineContext as EnhancedPipelineContext, QualityIssue, RecoveryAdvisor,
    RecoveryStrategy as EnhancedRecoveryStrategy, RelatedIssue, ResolutionStep, ResolutionStrategy,
    SeverityLevel, SimilarIssue, StackFrame, SuggestionEngine, SuggestionGenerator,
};
pub use enhanced_errors::{
    DataShape, EnhancedErrorBuilder, ErrorContext, ImpactLevel,
    PerformanceMetrics as EnhancedPerformanceMetrics, PerformanceWarningType, PipelineError,
    ResourceType, StructureErrorType, TypeViolationType,
};
pub use ensemble::{
    CompetenceEstimation, DynamicEnsembleSelector, DynamicEnsembleSelectorBuilder,
    DynamicEnsembleSelectorTrained, FusionStrategy, HierarchicalComposition,
    HierarchicalCompositionBuilder, HierarchicalCompositionTrained, HierarchicalNode,
    HierarchicalStrategy, ModelFusion, ModelFusionBuilder, ModelFusionTrained, SelectionStrategy,
    VotingClassifier, VotingClassifierBuilder, VotingClassifierTrained, VotingRegressor,
    VotingRegressorBuilder, VotingRegressorTrained,
};
pub use execution::{
    ExecutionEngineConfig, ExecutionStrategy, ExecutionTask,
    ParameterValue as ComposableParameterValue, PerformanceGoals, ResourceConstraints,
    StrategyConfig, StrategyMetrics, TaskHandle, TaskPriority, TaskResult, TaskScheduler,
    TaskStatus, TaskType,
};
pub use execution_hooks::{
    CustomHook, CustomHookBuilder, ExecutionContext as HookExecutionContext, ExecutionHook,
    HookData, HookManager, HookPhase, HookResult, LogLevel as HookLogLevel, LoggingHook,
    MemoryUsage as HookMemoryUsage, PerformanceHook, PerformanceMetrics as HookPerformanceMetrics,
    ValidationHook,
};
pub use external_integration::{
    AuthConfig, AuthCredentials, AuthType, BackoffStrategy, CircuitBreakerConfig,
    CircuitBreakerState, CircuitState, ConnectionConfig, DatabaseIntegration, ExternalIntegration,
    ExternalIntegrationManager, HealthCheckConfig, HealthStatus, IntegrationConfig,
    IntegrationData, IntegrationRequest, IntegrationResponse, IntegrationType, Operation,
    OperationResult, RateLimitConfig, RefreshConfig, RestApiIntegration, RetryCondition,
    RetryPolicy, TimeoutConfig, TlsConfig,
};
pub use feature_engineering::{
    AutoFeatureEngineer, ColumnType, ColumnTypeDetector, DetectionMethod, FeatureInteraction,
    FeatureInteractionDetector, InteractionType,
};
pub use few_shot::{
    DistanceMetric, FewShotLearnerType, FewShotPipeline, FewShotPipelineTrained, MAMLLearner,
    MAMLLearnerTrained, MetaLearnerWrapper, PrototypicalNetwork, PrototypicalNetworkTrained,
    SupportSet,
};
pub use fluent_api::{
    CacheStrategy, CachingConfiguration, DebugConfiguration, FeatureEngineeringChain,
    FeatureUnionBuilder, FluentPipelineBuilder, ImputationStrategy, LogLevel, MemoryConfiguration,
    PipelineConfiguration, PipelinePresets, PreprocessingChain, ValidationConfiguration,
    ValidationLevel,
};
pub use memory_optimization::{
    MemoryEfficientOps, MemoryMonitor, MemoryMonitorConfig, MemoryPool, MemoryStatistics,
    MemoryUsage as MemoryOptimizationUsage, PoolStatistics, StreamingBuffer,
};
pub use meta_learning::{
    AdaptationStrategy, Experience, ExperienceStorage, MetaLearningPipeline,
    MetaLearningPipelineTrained,
};
pub use middleware::{
    AlertManager, AlertRule, AlertSeverity as MiddlewareAlertSeverity, AuthenticationCredentials,
    AuthenticationMethod as MiddlewareAuthMethod, AuthenticationMiddleware, AuthenticationProvider,
    AuthorizationConfig as MiddlewareAuthConfig, AuthorizationMiddleware, CacheConfig, CacheEntry,
    CachingMiddleware, ErrorAction, MiddlewareChain, MiddlewareChainConfig, MiddlewareContext,
    MiddlewareStats, MonitoringMiddleware, PipelineMiddleware, TransformationMiddleware,
    UserInfo as MiddlewareUserInfo, ValidationMiddleware,
};
pub use mock::{MockPredictor, MockTransformer};
pub use modular_framework::{
    CapabilityMismatch, CompatibilityReport, ComponentCapability, ComponentConfig,
    ComponentDependency, ComponentFactory, ComponentInfo, ComponentMetadata, ComponentNode,
    ComponentRegistry, ComponentStatus, CompositionContext, ConfigValue as ModularConfigValue,
    DependencyGraph, EnvironmentSettings, ErrorHandlingStrategy, ExecutionCondition,
    ExecutionMetadata as ModularExecutionMetadata, ExecutionStrategy as ModularExecutionStrategy,
    LogLevel as ModularLogLevel, MissingDependency, ModularPipeline, ModularPipelineBuilder,
    PipelineConfig as ModularPipelineConfig, PipelineStep as ModularPipelineStep,
    PluggableComponent, ResourceLimits, ResourceManager, VersionConflict,
};
pub use monitoring::{
    Anomaly, AnomalySeverity, AnomalyType, ExecutionContext as MonitoringExecutionContext,
    ExecutionHandle, ExecutionStatus, Metric, MetricsSnapshot, MonitorConfig, PerformanceAnalysis,
    PerformanceBaseline, PerformanceTrends, PipelineMonitor, StagePerformance, Trend,
};
pub use nlp_pipelines::{
    AnalysisResult, BagOfWordsExtractor, ContextManager, ConversationResponse, ConversationTurn,
    ConversationalAI, DocumentParser, DocumentProcessor, Entity, EvaluationConfig,
    EvaluationMetric, FeatureExtractionConfig, FeatureExtractor as NLPFeatureExtractor,
    LanguageDetector, LanguageModel, ModelConfig as NLPModelConfig, ModelPrediction,
    ModelType as NLPModelType, MultiLanguageSupport, NERAnalyzer, NLPPipeline, NLPPipelineConfig,
    OutputFormatter, PreprocessingConfig, ProcessingResult, ProcessingStats,
    QuestionAnsweringModel, SentimentAnalyzer, SimpleLanguageModel, SummarizationStrategy,
    TextAnalyzer, TextClassifier, TextNormalizer, TextPreprocessor, TextSummarizationModel,
    TfIdfExtractor, TopicModelingAnalyzer, TrainingConfig as NLPTrainingConfig, TranslationModel,
    WordEmbeddingExtractor,
};
pub use optimization::{
    ErrorHandlingStrategy as OptimizationErrorHandlingStrategy, FallbackStrategy,
    MultiObjectiveResult, OptimizationResults, ParameterSpace, ParameterType, ParetoFront,
    PipelineOptimizer, PipelineValidator, RobustPipelineExecutor, ScoringMetric, SearchStrategy,
};
// pub use pattern_optimization::{
//     MultiObjectiveOptimizer, MOOAlgorithm, ParetoFrontManager,
//     ScalarizationMethod, ScalarizationType, PreferenceModel, PreferenceModelType,
//     PreferenceStructure, PreferenceRelation, RelationType, IterationResult,
//     SolutionSelector, DiversityMaintainer, MOOConvergenceDetector,
//     MOOPerformanceIndicators, ArchiveManager, DominationAnalyzer,
//     FrontExtractor, FrontQualityAssessor, FrontVisualizer,
// };
pub use parallel_execution::{
    AsyncTask, LoadBalancingStrategy, ParallelConfig, ParallelExecutionStrategy, ParallelExecutor,
    ParallelPipeline, ParallelTask, TaskResult as ParallelTaskResult, WorkerStatistics,
};
pub use performance_profiler::{
    BottleneckMetrics, BottleneckSeverity as ProfilerBottleneckSeverity,
    BottleneckType as ProfilerBottleneckType, ComparativeAnalysis, CpuSample as ProfilerCpuSample,
    GpuSample, ImplementationDifficulty, MemorySample as ProfilerMemorySample,
    OptimizationCategory, OptimizationHint, OptimizationPriority, OverallMetrics,
    PerformanceProfiler as ProfilerPerformanceProfiler,
    PerformanceReport as ProfilerPerformanceReport, ProfileSession, ProfilerConfig, StageProfile,
    SummaryMetrics, TrendDirection as ProfilerTrendDirection,
};
pub use performance_testing::{
    BenchmarkContext, BenchmarkResult as PerformanceBenchmarkResult, BenchmarkStorage,
    CpuStatistics, EnvironmentConfig as PerformanceEnvironmentConfig, EnvironmentMetadata,
    MemoryStatistics as PerformanceMemoryStatistics, OutlierDetection,
    PerformanceMetrics as PerformanceTestingMetrics, PerformanceRegressionTester,
    PerformanceReport, ProfilingConfig, RegressionAnalysis, RegressionSeverity,
    RegressionThresholds, StatisticalAnalysisConfig, StatisticalTest, SystemInfo,
    ThroughputMetrics, TimeStatistics, TrendAnalysis,
};
pub use pipeline::{Pipeline, PipelineBuilder, PipelinePredictor, PipelineStep, PipelineTrained};
pub use pipeline_visualization::{
    DataSpecification, DataType as VisualizationDataType, EdgeProperties, EdgeStyle, ExportFormat,
    FontProperties, FontWeight, GraphEdge, GraphNode, IoSpecification, LayoutAlgorithm, NodeShape,
    NodeSize, ParameterValue as VisualizationParameterValue, PipelineGraph, PipelineVisualizer,
    ShapeSpecification, VisualProperties, VisualizationConfig,
};
pub use plugin_architecture::{
    ComponentConfig as PluginComponentConfig, ComponentContext,
    ComponentFactory as PluginComponentFactory, ComponentSchema, ConfigValue as PluginConfigValue,
    ParameterConstraint, ParameterSchema, ParameterType as PluginParameterType, Plugin,
    PluginCapability, PluginComponent, PluginConfig, PluginContext, PluginEstimator, PluginLoader,
    PluginMetadata, PluginRegistry, PluginTransformer,
};
pub use profile_guided_optimization::{
    AccessPattern, CacheOptimizationHints, DataCharacteristics, ExecutionMetrics, HardwareContext,
    MLPerformancePredictor, MemoryLayout, OptimizationLevel, OptimizationStats,
    OptimizationStrategy, OptimizerConfig, ParallelStrategy, PerformancePredictor,
    PerformanceProfile, ProfileGuidedOptimizer, SimdFeature,
};
pub use property_testing::{
    PipelinePropertyTester, PropertyTestCase, PropertyTestGenerator, PropertyTestResult,
    StatisticalValidator, TestSuiteResult, TestSuiteRunner,
    ValidationResult as PropertyValidationResult, ValidationStatistics,
};
pub use quality_assurance::{
    AutomatedQualityAssurance, ComplianceStatus, ExecutiveSummary,
    IssueCategory as QAIssueCategory, IssueSeverity as QAIssueSeverity, QAConfig,
    QualityAssessment, QualityGates, QualityIssue as QAQualityIssue, QualityMetrics, QualityReport,
    QualityStandards, RecommendationCategory as QARecommendationCategory,
    RecommendationPriority as QARecommendationPriority, TrendDirection as QATrendDirection,
};
pub use quantum::{
    QuantumBackend, QuantumEnsemble, QuantumGate, QuantumPipeline, QuantumPipelineBuilder,
    QuantumPipelineStep, QuantumTransformer,
};
// pub use retry::{
//     AdaptiveLearningSystem, BackoffAlgorithm, ExponentialBackoffAlgorithm, RetryManager,
//     RetryStrategy, RetryContext, RetryConfig as RetryConfiguration, RetryError, RetryMetrics,
//     AdaptiveStrategy, CircuitBreakerStrategy, LinearBackoffStrategy, FeatureEngineering,
//     PolicyEvaluator, RetryPolicyEngine, ConfigurationManager, GlobalRetryConfig,
//     AlertingSystem, RetryMetricsCollector, ModelPerformanceMetrics, SystemStatistics,
// };
pub use scheduling::{
    ResourcePool, RetryConfig, ScheduledTask, SchedulerStatistics, SchedulingStrategy,
    TaskScheduler as SchedulingTaskScheduler, TaskState, Workflow, WorkflowManager,
};
pub use simd_optimizations::{SimdConfig, SimdDataLayout, SimdFeatureOps, SimdOps};
pub use state_management::{
    CheckpointConfig, ExecutionStatistics, PersistenceStrategy, PipelineVersionControl, StateData,
    StateManager, StateSnapshot, StateSynchronizer,
};
pub use streaming::{
    StateManagement, StreamConfig, StreamDataPoint, StreamStats, StreamWindow, StreamingPipeline,
    StreamingPipelineTrained, UpdateStrategy, WindowingStrategy,
};
pub use stress_testing::{
    ComputationType, EdgeCase, MemoryPattern, ResourceMonitor, ResourceStats, ResourceUsageStats,
    StressTestConfig, StressTestIssue, StressTestReport, StressTestResult, StressTestScenario,
    StressTester,
};
// TODO: Enable once time_series_pipelines modules are implemented
// pub use time_series_pipelines::{
//     AnomalyDetector, AnomalyType as TimeSeriesAnomalyType, DataStream, EdgeNode, FeatureEngineer,
//     IoTConfig, IoTDataPipeline, IoTDevice, MessageBroker, PostProcessor as TSPostProcessor,
//     QualityControlConfig, RealTimeConfig, SamplingFrequency, SeasonalPattern,
//     TimeSeriesCharacteristics, TimeSeriesConfig as TSConfig, TimeSeriesMetrics, TimeSeriesModel,
//     TimeSeriesPipeline, TimeSeriesTransform, TrendType,
// };
pub use enhanced_wasm_integration::{
    BrowserFeature, BrowserFeatureDetection, BrowserInfo,
    BrowserIntegration as EnhancedBrowserIntegration, CompilationTarget, CompiledWasmModule,
    CpuIntensity, ExecutionContext as WasmExecutionContext, ExecutionProfile,
    FeatureDetectionStrategy, FunctionHandle, FunctionSignature, GeneratedBinding, IoProfile,
    JsBindingsGenerator, JsType, LoadedWasmModule, MemoryConstraints,
    MemoryLayout as WasmMemoryLayout, MemoryPermissions, MemoryProfile, MemoryRegion, ModuleLoader,
    ModuleSource, OptimizationPass, OptimizationResult,
    OptimizationStrategy as WasmOptimizationStrategy, PerformanceHints,
    PerformanceProfile as WasmPerformanceProfile, PerformanceRequirements, ProfilingSession,
    ScalingProfile, TaskData, TaskPriority as WasmTaskPriority, TaskType as WasmTaskType,
    TypedArrayType, WasmArchitecture, WasmCompiler as EnhancedWasmCompiler, WasmExport,
    WasmExportValue, WasmImport, WasmInstance, WasmIntegrationConfig, WasmIntegrationManager,
    WasmMemoryView, WasmModuleManager, WasmModuleMetadata, WasmPerformanceOptimizer, WasmProfiler,
    WasmType, WasmValue as EnhancedWasmValue, WebApiIntegration,
    WorkerStatistics as WasmWorkerStatistics, WorkerStatus, WorkerTask, WorkerThread,
    WorkerThreadManager,
};
pub use transfer_learning::{
    domain_adaptation::{
        DomainAdaptationPipeline, DomainAdaptationPipelineTrained, DomainAdaptationStrategy,
    },
    AdaptationConfig, LearningRateSchedule, PretrainedModel, TransferLearningPipeline,
    TransferLearningPipelineTrained, TransferStrategy,
};
pub use type_safety::{
    CategoricalInput, ClassificationOutput, DataFlowValidation, DataFlowValidator, DenseOutput,
    Input, MixedInput, NumericInput, Output, PipelineValidation, PipelineValidationError,
    RegressionOutput, SparseOutput, StructureValidation, TypeCompatible, TypedEstimator,
    TypedFeatureUnion, TypedPipelineBuilder, TypedPipelineStage, TypedTransformer,
};
pub use validation::{
    ComprehensivePipelineValidator, CrossValidationResult, CrossValidator, DataValidationResult,
    DataValidator, MessageLevel, PerformanceValidationResult, PerformanceValidator,
    RobustnessTestResult, RobustnessTester, StatisticalValidationResult,
    StatisticalValidator as ComprehensiveStatisticalValidator, StructureValidationResult,
    StructureValidator, ValidationMessage, ValidationReport as ComprehensiveValidationReport,
};
pub use wasm_integration::{
    BrowserIntegration, DataSchema, OptimizationLevel as WasmOptimizationLevel, PipelineMetadata,
    WasmCompiler, WasmConfig, WasmDataType, WasmModule, WasmOptimization, WasmPipeline, WasmStep,
    WasmStepType, WasmValue,
};
#[cfg(feature = "workflow")]
pub use workflow_language::{
    CodeLanguage, ComponentRegistry as WorkflowComponentRegistry, DataType, DslError, DslLexer,
    DslParser, ExecutionConfig as WorkflowExecutionConfig, ExecutionMode, FileFormat,
    ParameterValue, PipelineDSL, StepDefinition, StepType as WorkflowStepType, Token,
    VisualPipelineBuilder, WorkflowDefinition, WorkflowMetadata,
};
pub use zero_cost::{
    AllocationInfo, Arena, AtomicRcData, ConcurrencyStats, CowData, LockFreeQueue,
    MemoryLeakConfig, MemoryLeakDetector, MemoryPool as ZeroCostMemoryPool, MemoryStats,
    PooledBuffer, QueueStats, SafeConcurrentData, SharedData, TrackedAllocation, WeakRcData,
    WorkStealingDeque, WorkStealingStats, ZeroCopySlice, ZeroCopyView, ZeroCostBuffer,
    ZeroCostBuilder, ZeroCostCompose, ZeroCostComposition, ZeroCostConditional, ZeroCostEstimator,
    ZeroCostFeatureSelector, ZeroCostFeatureUnion, ZeroCostLayout, ZeroCostParallel,
    ZeroCostPipeline, ZeroCostStep,
};

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::types::Float;
use sklears_core::{
    error::Result as SklResult,
    prelude::{Fit as CoreFit, Predict, SklearsError, Transform},
    traits::{Estimator, Fit, Untrained},
};
use std::collections::HashMap;

/// Transformed Target Regressor
///
/// Meta-estimator that transforms the target y before fitting a regression model.
/// The predictions are mapped back to the original space via an inverse transform.
///
/// # Parameters
///
/// * `regressor` - The regressor to use for prediction
/// * `transformer` - The transformer to apply to the target
/// * `func` - Function to transform the target
/// * `inverse_func` - Function to inverse transform the predictions
/// * `check_inverse` - Whether to check the inverse transform
///
// Type aliases to reduce complexity
type TransformBox = Box<dyn for<'a> Transform<ArrayView1<'a, Float>, Array1<f64>> + Send + Sync>;
type TransformFunc = fn(&ArrayView1<'_, Float>) -> Array1<f64>;

/// # Examples
///
/// ```
/// use sklears_compose::TransformedTargetRegressor;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0], [2.0], [3.0]];
/// let y = array![1.0, 4.0, 9.0];
/// ```
pub struct TransformedTargetRegressor<S = Untrained> {
    state: S,
    regressor: Option<Box<dyn PipelinePredictor>>,
    transformer: Option<TransformBox>,
    func: Option<TransformFunc>,
    inverse_func: Option<TransformFunc>,
    check_inverse: bool,
}

/// Trained state for `TransformedTargetRegressor`
pub struct TransformedTargetRegressorTrained {
    fitted_regressor: Box<dyn PipelinePredictor>,
    fitted_transformer: Option<TransformBox>,
    func: Option<TransformFunc>,
    inverse_func: Option<TransformFunc>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl TransformedTargetRegressor<Untrained> {
    /// Create a new `TransformedTargetRegressor`
    #[must_use]
    pub fn new(regressor: Box<dyn PipelinePredictor>) -> Self {
        Self {
            state: Untrained,
            regressor: Some(regressor),
            transformer: None,
            func: None,
            inverse_func: None,
            check_inverse: true,
        }
    }

    /// Set the transformer
    #[must_use]
    pub fn transformer(
        mut self,
        transformer: Box<dyn for<'a> Transform<ArrayView1<'a, Float>, Array1<f64>> + Send + Sync>,
    ) -> Self {
        self.transformer = Some(transformer);
        self
    }

    /// Set transformation function
    pub fn func(mut self, func: fn(&ArrayView1<'_, Float>) -> Array1<f64>) -> Self {
        self.func = Some(func);
        self
    }

    /// Set inverse transformation function
    pub fn inverse_func(mut self, inverse_func: fn(&ArrayView1<'_, Float>) -> Array1<f64>) -> Self {
        self.inverse_func = Some(inverse_func);
        self
    }

    /// Set whether to check inverse transform
    #[must_use]
    pub fn check_inverse(mut self, check: bool) -> Self {
        self.check_inverse = check;
        self
    }
}

impl Estimator for TransformedTargetRegressor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>>
    for TransformedTargetRegressor<Untrained>
{
    type Fitted = TransformedTargetRegressor<TransformedTargetRegressorTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        if let Some(y_values) = y.as_ref() {
            let mut regressor = self.regressor.ok_or_else(|| SklearsError::InvalidData {
                reason: "No regressor provided".to_string(),
            })?;

            // Transform the target if transformer or function is provided
            let transformed_y = if let Some(ref transformer) = self.transformer {
                transformer.transform(y_values)?
            } else if let Some(func) = self.func {
                func(y_values)
            } else {
                y_values.mapv(|v| v)
            };

            // Fit the regressor on transformed target
            regressor.fit(x, &transformed_y.view())?;

            Ok(TransformedTargetRegressor {
                state: TransformedTargetRegressorTrained {
                    fitted_regressor: regressor,
                    fitted_transformer: self.transformer,
                    func: self.func,
                    inverse_func: self.inverse_func,
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                },
                regressor: None,
                transformer: None,
                func: None,
                inverse_func: None,
                check_inverse: self.check_inverse,
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Target values required for fitting".to_string(),
            ))
        }
    }
}

impl TransformedTargetRegressor<TransformedTargetRegressorTrained> {
    /// Predict using the fitted regressor and inverse transform
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        // Get predictions from the fitted regressor
        let transformed_predictions = self.state.fitted_regressor.predict(x)?;

        // Apply inverse transformation
        let predictions = if let Some(inverse_func) = self.state.inverse_func {
            inverse_func(&transformed_predictions.view())
        } else if let Some(ref transformer) = self.state.fitted_transformer {
            // Note: This assumes the transformer has an inverse_transform method
            // In a real implementation, you'd need a proper inverse transformer trait
            transformed_predictions
        } else {
            transformed_predictions
        };

        Ok(predictions)
    }

    /// Get the fitted regressor
    #[must_use]
    pub fn regressor(&self) -> &dyn PipelinePredictor {
        &*self.state.fitted_regressor
    }
}

/// Feature Union for parallel feature extraction
///
/// Applies multiple transformers in parallel and concatenates their results.
///
/// # Examples
///
/// ```ignore
/// use sklears_compose::FeatureUnion;
/// use scirs2_core::ndarray::array;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0]];
/// let union = FeatureUnion::new()
///     .transformer("trans1", Box::new(MockTransformer::new()))
///     .transformer("trans2", Box::new(MockTransformer::new()));
/// ```
#[derive(Debug)]
pub struct FeatureUnion<S = Untrained> {
    state: S,
    transformers: Vec<(String, Box<dyn PipelineStep>)>,
    n_jobs: Option<i32>,
    transformer_weights: Option<HashMap<String, f64>>,
    preserve_dataframe: bool,
}

/// Trained state for `FeatureUnion`
#[derive(Debug)]
pub struct FeatureUnionTrained {
    fitted_transformers: Vec<(String, Box<dyn PipelineStep>)>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl FeatureUnion<Untrained> {
    /// Create a new `FeatureUnion`
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            transformers: Vec::new(),
            n_jobs: None,
            transformer_weights: None,
            preserve_dataframe: false,
        }
    }

    /// Add a transformer
    #[must_use]
    pub fn transformer(mut self, name: &str, transformer: Box<dyn PipelineStep>) -> Self {
        self.transformers.push((name.to_string(), transformer));
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set transformer weights
    #[must_use]
    pub fn transformer_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.transformer_weights = Some(weights);
        self
    }

    /// Set preserve dataframe option
    #[must_use]
    pub fn preserve_dataframe(mut self, preserve: bool) -> Self {
        self.preserve_dataframe = preserve;
        self
    }
}

impl Default for FeatureUnion<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for FeatureUnion<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for FeatureUnion<Untrained> {
    type Fitted = FeatureUnion<FeatureUnionTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let mut fitted_transformers = Vec::new();

        for (name, mut transformer) in self.transformers {
            transformer.fit(x, y.as_ref().copied())?;
            fitted_transformers.push((name, transformer));
        }

        Ok(FeatureUnion {
            state: FeatureUnionTrained {
                fitted_transformers,
                n_features_in: x.ncols(),
                feature_names_in: None,
            },
            transformers: Vec::new(),
            n_jobs: self.n_jobs,
            transformer_weights: self.transformer_weights,
            preserve_dataframe: self.preserve_dataframe,
        })
    }
}

impl FeatureUnion<FeatureUnionTrained> {
    /// Transform data using all fitted transformers and concatenate results
    pub fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        if self.state.fitted_transformers.is_empty() {
            return Ok(x.mapv(|v| v));
        }

        let mut results = Vec::new();

        for (name, transformer) in &self.state.fitted_transformers {
            let mut transformed = transformer.transform(x)?;

            // Apply weights if specified
            if let Some(ref weights) = self.transformer_weights {
                if let Some(&weight) = weights.get(name) {
                    transformed.mapv_inplace(|v| v * weight);
                }
            }

            results.push(transformed);
        }

        // Concatenate all results along the feature axis
        if results.len() == 1 {
            Ok(results.into_iter().next().unwrap())
        } else {
            let total_features: usize = results
                .iter()
                .map(scirs2_core::ndarray::ArrayBase::ncols)
                .sum();
            let n_samples = results[0].nrows();

            let mut concatenated = Array2::zeros((n_samples, total_features));
            let mut col_idx = 0;

            for result in results {
                let end_idx = col_idx + result.ncols();
                concatenated
                    .slice_mut(s![.., col_idx..end_idx])
                    .assign(&result);
                col_idx = end_idx;
            }

            Ok(concatenated)
        }
    }

    /// Get the fitted transformers
    #[must_use]
    pub fn transformers(&self) -> &[(String, Box<dyn PipelineStep>)] {
        &self.state.fitted_transformers
    }
}

// Import ndarray slice macro
use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mock_transformer() {
        let transformer = MockTransformer::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let result = crate::PipelineStep::transform(&transformer, &x.view()).unwrap();
        assert_eq!(result, x.mapv(|v| v as f64));
    }

    #[test]
    fn test_mock_predictor() {
        let mut predictor = MockPredictor::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        predictor.fit(&x.view(), &y.view()).unwrap();
        assert!(predictor.is_fitted());

        let predictions = predictor.predict(&x.view()).unwrap();
        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_feature_union() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let union = FeatureUnion::new()
            .transformer("trans1", Box::new(MockTransformer::new()))
            .transformer("trans2", Box::new(MockTransformer::with_scale(2.0)));

        let fitted_union = union.fit(&x.view(), &None).unwrap();
        let result = fitted_union.transform(&x.view()).unwrap();

        // Should concatenate results from both transformers
        assert_eq!(result.ncols(), 4); // 2 features * 2 transformers
        assert_eq!(result.nrows(), 2); // Same number of samples
    }

    #[test]
    fn test_pipeline_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let pipeline = Pipeline::builder()
            .step("scaler", Box::new(MockTransformer::new()))
            .estimator(Box::new(MockPredictor::new()))
            .build();

        let fitted_pipeline = pipeline.fit(&x.view(), &Some(&y.view())).unwrap();
        let predictions = fitted_pipeline.predict(&x.view()).unwrap();

        assert_eq!(predictions.len(), x.nrows());
    }
}
