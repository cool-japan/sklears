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
//! Model inspection and interpretation tools
//!
//! This crate provides comprehensive tools for understanding and interpreting machine learning models,
//! including feature importance, partial dependence plots, SHAP values, counterfactual explanations,
//! anchors, model complexity analysis, and model-agnostic explanations.

// #![warn(missing_docs)]

// Re-export core types and traits
pub use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

// Module declarations
pub mod adversarial;
pub mod anchors;
pub mod attention;
pub mod benchmarking;
pub mod builder;
pub mod causal;
pub mod complexity;
pub mod computer_vision;
pub mod counterfactual;
pub mod dashboard;
pub mod deep_learning;
pub mod distributed;
pub mod enterprise;
pub mod external_visualizations;
pub mod fairness;
pub mod federated;
pub mod framework;
pub mod gnn;
pub mod gpu;
pub mod hooks;
pub mod information_theoretic;
pub mod lazy;
pub mod llm;
pub mod local_explanations;
pub mod memory;
pub mod metrics_registry;
pub mod model_agnostic;
pub mod model_comparison;
pub mod multimodal;
pub mod nlp;
pub mod occlusion;
pub mod parallel;
pub mod partial_dependence;
pub mod permutation;
pub mod perturbation;
pub mod plugins;
pub mod probabilistic;
pub mod profiling;
pub mod quantum;
pub mod reporting;
pub mod rules;
pub mod sensitivity;
pub mod serialization;
pub mod shapley;
pub mod streaming;
pub mod testing;
pub mod time_series;
pub mod types;
pub mod uncertainty;
pub mod validation;
pub mod visualization;
pub mod visualization_backend;
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export commonly used types
pub use types::*;

// Re-export type safety features
pub use types::{
    explanation_methods, explanation_states, ExplanationConfig, ExplanationConstraint,
    ExplanationMethodValidator, ExplanationProperties, ExplanationValidator,
    FeatureImportanceConstraint, FixedSizeExplanation, GlobalGradientConfig,
    GlobalModelAgnosticConfig, GradientCompatible, LimeCompatible, LocalGradientConfig,
    LocalModelAgnosticConfig, ModelIntrospectable, ShapCompatible, ShapConstraint,
    TypedExplanation,
};

// Re-export main functions
pub use adversarial::{
    analyze_explanation_stability, compute_certified_robustness, generate_adversarial_examples,
    test_explanation_robustness, AdversarialAttack, AdversarialConfig, AdversarialExampleResult,
    CertificationMethod, CertifiedRobustnessResult, ExplanationRobustnessResult, RobustnessMetric,
    StabilityAnalysisResult, StabilityTrend, VerificationStatus,
};
pub use anchors::{explain_with_anchors, AnchorsConfig};
pub use attention::{
    analyze_attention, compute_gradcam, dissect_network, maximize_activation, visualize_features,
    ActivationMaximizationResult, AttentionAnalyzer, AttentionConfig, AttentionResult,
    AttentionType, FeatureVisualizationResult, GradCAMConfig,
    GradCAMResult as AttentionGradCAMResult, NetworkDissectionResult,
};
pub use benchmarking::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkReport, BenchmarkResult, BenchmarkingSuite,
    CategorySummary, InsightSeverity, InsightType, MemoryStatistics, PerformanceInsight,
    ProblemType, QualityMetrics, ReferenceComparison, TestConfiguration, TimingStatistics,
};
pub use builder::{
    ComparisonResult, ComparisonResults, ComparisonStudy, ComparisonStudyBuilder,
    CounterfactualConfig as BuilderCounterfactualConfig, ExplanationBuilder,
    ExplanationPipelineExecutor, FeatureSelection, LimeConfig, OptimizationMethod,
    PermutationConfig as BuilderPermutationConfig, PipelineBuilder, PipelineExecutionResult,
    PipelineStep, ScoreFunction as BuilderScoreFunction, ShapConfig, StepMetadata,
};
pub use causal::{
    analyze_instrumental_variables, analyze_mediation, apply_do_calculus,
    discover_causal_structure, estimate_causal_effect, CausalConfig, CausalEffectMethod,
    CausalEffectResult, CausalGraph, InstrumentalVariableResult, MediationResult,
};
pub use complexity::{analyze_model_complexity, ComplexityConfig};
pub use computer_vision::{
    explain_image_with_lime, explain_object_detection, explain_segmentation, generate_gradcam,
    generate_saliency_map, ComputerVisionConfig, DetectedObject, GradCAMResult, GradCAMStats,
    Image, ImageLimeResult, KeyFeature, ObjectDetectionExplanation, SaliencyMapResult,
    SaliencyMethod, SegmentExplanation, SegmentationExplanation, Superpixel,
};
pub use counterfactual::{
    generate_actionable_counterfactual, generate_causal_counterfactual, generate_counterfactual,
    generate_diverse_counterfactuals, generate_feasible_counterfactual,
    generate_nearest_counterfactual, CounterfactualConfig, FeasibilityConfig,
};
pub use dashboard::{
    Alert, AlertSeverity, AlertThresholds, AlertType, Dashboard, DashboardConfig,
    DashboardDataPoint, DashboardLayout, DashboardState, DashboardTheme, DashboardUpdate,
    WidgetConfig, WidgetType,
};
pub use deep_learning::{
    ConceptActivationVector, ConceptDatabase, ConceptDiscoveryMethod, ConceptHierarchy,
    DeepLearningAnalyzer, DeepLearningConfig, DetectedConcept, DisentanglementMetrics,
    NetworkDissectionResult as DeepNetworkDissectionResult, TCAVResult,
};
pub use distributed::{
    ClusterConfig, ClusterExplanationOrchestrator, ClusterHealth, ClusterStatistics,
    DistributedCoordinator, DistributedTask, HealthStatus, LoadBalancingStrategy, TaskResult,
    TaskStatus, TaskType, WorkerConfig, WorkerNode,
};
pub use enterprise::{
    AccessControl, AccessControlConfig, AccessLevel, ActionItem, AlertRule, AuditEvent,
    AuditEventType, AuditLogger, AuditRecord, AuditSeverity, AuditTrail, ComplianceFramework,
    ComplianceReport, ComplianceReporter, ComplianceRule, ComplianceStatus, EnterpriseConfig,
    ExplanationLineage, ExplanationQualityMonitor, LineageNode, LineageRelation, LineageTracker,
    OperationType, Permission, PermissionSet, QualityAlert, QualityDataPoint, QualityLevel,
    QualityMetric, QualityMonitorConfig, QualityStatistics, QualityThreshold, QualityTrend,
    QualityTrendDirection, RegulatoryRequirement, RiskLevel, Role, RoleManager,
    SecureExplanationExecutor, SecurityContext, User, UserGroup,
};
pub use external_visualizations::{
    D3Backend, D3Config, PlotlyBackend, PlotlyConfig, PlotlyPlotConfig, VegaLiteBackend,
    VegaLiteConfig, VegaLiteDefaultConfig,
};
pub use fairness::{
    analyze_demographic_parity, analyze_equalized_odds, analyze_individual_fairness,
    assess_fairness, compute_group_fairness_metrics, detect_bias, BiasDetectionResult,
    DemographicParityResult, DistanceMetric, EqualizedOddsResult, FairnessConfig, FairnessMetrics,
    FairnessResult, IndividualFairnessResult,
};
pub use federated::{
    AggregationStats, FederatedConfig, FederatedExplainer, FederatedExplanation, PrivacyMechanism,
};
pub use framework::{
    ChainedConfig, ChainedStrategy, CombinedConfig, CombinedOutput, CombinedStrategy,
    CounterfactualExplainer, Explainer, ExplanationMetadata, ExplanationPipeline,
    ExplanationPostProcessor, ExplanationStrategy, FeatureAttributor, FeatureExplainer,
    GlobalExplainer, GradientAttributor, LocalExplainer, PipelineResult, UncertainExplanation,
    UncertaintyAwareExplainer, ValidationResult, ValidationViolation, ViolationSeverity,
};
pub use gnn::{
    GNNExplainer, GNNExplainerConfig, GNNExplanation, GNNTask, Graph, MessagePassingExplanation,
};
pub use gpu::{
    utils as gpu_utils, GpuBackend, GpuBuffer, GpuConfig, GpuContext, GpuDevice,
    GpuExplanationComputer, GpuPerformanceStats,
};
pub use hooks::{
    CustomEvent, ErrorInfo, ExecutionMetrics, ExplanationHook, HookContext, HookEvent,
    HookRegistry, HookRegistryStatistics, HookResult, HookedExplanationExecutor, LogLevel,
    LoggingHook, MemoryInfo, MetricsHook, ProgressInfo, TimingInfo,
};
pub use information_theoretic::{
    analyze_information_bottleneck, analyze_mutual_information, apply_minimum_description_length,
    compute_information_gain_attribution, generate_entropy_explanations, EntropyExplanationResult,
    EstimationMethod, InformationBottleneckResult, InformationGainResult,
    InformationTheoreticConfig, MDLResult, MutualInformationResult,
};
pub use lazy::{
    LazyComputationManager, LazyConfig, LazyExecutionStats, LazyExplanation,
    LazyExplanationPipeline, LazyFeatureImportance, LazyShapValues,
};
pub use llm::{
    CounterfactualText, LLMAttentionExplanation, LLMExplainer, LLMExplainerConfig, LLMExplanation,
    LLMTask, LayerImportance, LayerType, NeuronActivation, PromptSensitivity, PromptVariation,
    TokenizedInput, VariationType,
};
pub use local_explanations::{explain_locally, LocalExplanationConfig, LocalExplanationMethod};
pub use memory::{
    cache_friendly_permutation_importance, cache_friendly_shap_computation, CacheConfig, CacheKey,
    CacheStatistics, ExplanationCache, ExplanationDataLayout, MemoryLayoutManager,
};
pub use metrics_registry::{
    CompletenessMetric, ComputationMetadata, ComputationalComplexity, ExplanationData,
    ExplanationMetric, ExplanationType, FidelityMetric, MetricCategory, MetricInput,
    MetricMetadata, MetricOutput, MetricProperties, MetricRegistry, RegistryStatistics,
    StabilityMetric,
};
pub use model_agnostic::{explain_model_agnostic, ModelAgnosticConfig};
pub use model_comparison::{
    analyze_ensemble, assess_prediction_stability, compare_models, compute_model_agreement,
    BiasVarianceDecomposition, DiversityMetrics, EnsembleAnalysisResult, EnsembleMethod,
    ModelComparisonConfig, ModelComparisonResult, ModelMetrics, StabilityMetrics,
};
pub use multimodal::{
    CrossModalAttention, FusionExplanation, FusionStrategy, InteractionType, ModalityContributions,
    ModalityInteraction, ModalityType, MultiModalConfig, MultiModalExplainer,
    MultiModalExplanation, MultiModalInput,
};
pub use nlp::{
    analyze_text_attention, explain_semantic_similarity, explain_syntax, explain_text_with_lime,
    visualize_word_importance, AttentionExplanation, AttentionPattern, AttentionPatternType,
    NLPConfig, SemanticCluster, SemanticExplanation, SemanticRelation, SyntacticExplanation,
    SyntacticNode, SyntacticPattern, SyntacticTree, Token, WordImportanceResult,
};
pub use occlusion::{analyze_occlusion, OcclusionConfig, OcclusionMethod};
#[cfg(feature = "parallel")]
pub use parallel::{
    compute_shap_parallel, process_batches_optimized, process_batches_parallel,
    AdaptiveBatchConfig, BatchConfig, BatchStats, CacheAwareExplanationStore, CompressedBatch,
    HighPerformanceBatchProcessor, MemoryPool, ParallelConfig, ParallelExplanation,
    ParallelPermutationImportance, PermutationInput, ProgressCallback, StreamingBatchProcessor,
};
pub use partial_dependence::partial_dependence;
pub use permutation::permutation_importance;
pub use perturbation::{
    analyze_robustness, generate_perturbations, PerturbationConfig, PerturbationStrategy,
};
pub use plugins::{
    ExampleCustomPlugin, ExecutionMetadata, ExecutionStatistics, ExplanationPlugin, InputType,
    LogLevel as PluginLogLevel, OutputType, PluginCapabilities, PluginCapabilityFilter,
    PluginConfig, PluginData, PluginExecution, PluginInput, PluginManager, PluginMetadata,
    PluginOutput, PluginOutputData, PluginParameter, PluginRegistry, PluginRegistryStatistics,
};
pub use probabilistic::{
    bayesian_model_averaging, generate_bayesian_explanation,
    generate_probabilistic_counterfactuals, quantify_explanation_uncertainty,
    BayesianExplanationResult, BayesianModelAveragingResult, ProbabilisticConfig,
    ProbabilisticCounterfactualResult, UncertainExplanationResult,
};
pub use profiling::{
    BottleneckType, HotPath, MethodProfile, OptimizationComplexity, OptimizationConfig,
    OptimizationOpportunity, OptimizationType, PerformanceBottleneck, ProfileGuidedOptimizer,
    RuntimeStatistics, Severity,
};
pub use quantum::{
    CircuitComplexity, GateInstance, QuantumCircuit, QuantumExplainer, QuantumExplainerConfig,
    QuantumExplanation, QuantumGate,
};
pub use reporting::{
    export_report, generate_quick_report, CaveatsAndRecommendations, DatasetInfo,
    EthicalConsiderations, EvaluationData, ExplanationSummary, Factor, IntendedUse,
    InterpretabilityReport, InterpretabilityScorecard, ModelCard, ModelDetails, OutputFormat,
    QuantitativeAnalysis, ReportConfig, ReportGenerator, ReportMetadata, ReportModelMetrics,
    TemplateStyle, TrainingData,
};
pub use rules::{
    extract_rules_from_model, generate_decision_rules, generate_logical_explanations,
    mine_association_rules, simplify_rules, AssociationRule, ComparisonOperator, DecisionRule,
    RuleCondition, RuleExtractionConfig,
};
pub use sensitivity::{analyze_sensitivity, SensitivityConfig, SensitivityMethod};
pub use serialization::{
    BatchSummary, CompressionType, DataStatistics, DatasetMetadata, ExplanationBatch,
    ExplanationConfiguration, ModelMetadata, SerializableExplanationResult, SerializationConfig,
    SerializationFormat, SerializationSummary,
};
pub use shapley::{
    compute_deep_shap, compute_kernel_shap, compute_linear_shap, compute_partition_shap,
    compute_tree_shap, FeaturePartition, ShapleyConfig, ShapleyMetadata, ShapleyMethod,
    ShapleyResult, Tree, TreeNode,
};
pub use streaming::{
    create_data_chunks, BackgroundStatistics, OnlineAggregator, StreamingConfig,
    StreamingDataIterator, StreamingExplainer, StreamingExplanationResult, StreamingShapExplainer,
    StreamingStatistics,
};
pub use testing::{
    validate_explanation_output, ConsistencyTestConfig, FidelityTestConfig, PropertyTestConfig,
    PropertyTestResult, PropertyViolation, RobustnessTestConfig, TestingSuite,
};
pub use time_series::{
    AlignmentSegment, AlignmentType, DTWExplanation, DecompositionMethod, LagImportance,
    SeasonalDecomposition, TemporalImportance, TimeSeriesAnalyzer, TimeSeriesConfig, TrendAnalysis,
    TrendDirection, TrendSegment,
};
pub use uncertainty::{
    analyze_prediction_uncertainty, calibrate_predictions, compute_calibration_metrics,
    quantify_uncertainty, CalibratedModel, CalibrationMethod, CalibrationMetrics,
    UncertaintyAnalysis, UncertaintyConfig, UncertaintyEstimator, UncertaintyResult,
    UncertaintyType,
};
pub use validation::{
    AutomatedTestingResult, CaseStudyMetadata, CaseStudyValidationResult,
    ConsistencyValidationResult, DatasetValidationResult, FeedbackSeverity, HumanEvaluationResult,
    MethodAgreement, PerformanceBenchmark, QualitativeFeedback, StatisticalTest,
    SyntheticValidationResult, TestResult, ValidationConfig, ValidationFramework,
};
pub use visualization::{
    create_3d_plot, create_3d_shap_plot, create_3d_surface_plot, create_comparative_plot,
    create_feature_importance_plot, create_partial_dependence_plot, create_shap_visualization,
    Animation3D, ColorScheme, ComparativePlot, ComparisonType, EasingType, FeatureImportancePlot,
    FeatureImportanceType, MobileConfig, PartialDependencePlot, Plot3D, Plot3DType, PlotConfig,
    ShapPlot, ShapPlotType, Surface3D,
};
pub use visualization_backend::{
    AsciiBackend, BackendCapabilities, BackendConfig, BackendRegistry, ComparativeData,
    CustomPlotData, FeatureImportanceData, HtmlBackend, JsonBackend, PartialDependenceData,
    PlotType, RenderedVisualization, ShapData, Theme, VisualizationBackend, VisualizationMetadata,
    VisualizationRenderer,
};
// Temporarily disabled due to compilation issues
// #[cfg(feature = "wasm")]
// pub use wasm::{
//     BrowserCapabilities, JsInterop, WasmConfig, WasmExplainer, WasmMemoryManager, WasmShapComputer,
//     WasmSupportLevel, WasmVisualizer, WebGlSupport,
// };

// Additional utility functions and re-exports would go here
// (The rest of the original lib.rs content like SHAP, ICE plots, etc.)

// For now, let's include a few key functions from the original lib.rs
// that weren't moved to modules yet

// ✅ SciRS2 Policy Compliant Imports
use scirs2_core::ndarray::{ArrayView1, ArrayView2};

// Re-export the compute_score function from permutation module
pub use permutation::compute_score;

/// Feature Importance Data
///
/// This function creates feature importance data suitable for plotting from permutation importance results.
pub fn create_feature_importance_data(
    result: &PermutationImportanceResult,
    feature_names: Option<Vec<String>>,
) -> FeatureImportance {
    let feature_indices: Vec<usize> = (0..result.importances_mean.len()).collect();

    FeatureImportance::new(
        feature_indices,
        result.importances_mean.clone(),
        Some(result.importances_std.clone()),
        feature_names,
    )
}

/// Utility function to compute feature statistics
pub fn compute_feature_statistics(X: &ArrayView2<Float>) -> Vec<(Float, Float, Float, Float)> {
    let mut stats = Vec::new();

    for col_idx in 0..X.ncols() {
        let column = X.column(col_idx);
        let mean = column.mean().unwrap_or(0.0);

        let mut sorted_values: Vec<Float> = column.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_val = sorted_values.first().copied().unwrap_or(0.0);
        let max_val = sorted_values.last().copied().unwrap_or(0.0);

        let variance =
            column.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / column.len() as Float;
        let std_dev = variance.sqrt();

        stats.push((mean, std_dev, min_val, max_val));
    }

    stats
}

/// Utility function to validate input dimensions
pub fn validate_input_dimensions(X: &ArrayView2<Float>, y: &ArrayView1<Float>) -> SklResult<()> {
    if X.nrows() != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    if X.nrows() == 0 {
        return Err(SklearsError::InvalidInput(
            "Input data cannot be empty".to_string(),
        ));
    }

    if X.ncols() == 0 {
        return Err(SklearsError::InvalidInput(
            "Input data must have at least one feature".to_string(),
        ));
    }

    Ok(())
}

/// Utility function to generate feature names if not provided
pub fn generate_feature_names(n_features: usize) -> Vec<String> {
    (0..n_features).map(|i| format!("feature_{}", i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature_importance_plot_creation() {
        let result = PermutationImportanceResult {
            importances: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            importances_mean: array![0.15, 0.35],
            importances_std: array![0.05, 0.05],
        };

        let feature_data = create_feature_importance_data(&result, None);
        assert_eq!(feature_data.importances.len(), 2);
        assert_eq!(feature_data.feature_indices.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_feature_statistics() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let stats = compute_feature_statistics(&X.view());

        assert_eq!(stats.len(), 2);
        // First column: mean=3.0, min=1.0, max=5.0
        assert_eq!(stats[0].0, 3.0); // mean
        assert_eq!(stats[0].2, 1.0); // min
        assert_eq!(stats[0].3, 5.0); // max
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_input_validation() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        assert!(validate_input_dimensions(&X.view(), &y.view()).is_ok());

        let y_wrong = array![1.0]; // Wrong length
        assert!(validate_input_dimensions(&X.view(), &y_wrong.view()).is_err());
    }

    #[test]
    fn test_feature_name_generation() {
        let names = generate_feature_names(3);
        assert_eq!(names, vec!["feature_0", "feature_1", "feature_2"]);
    }
}
