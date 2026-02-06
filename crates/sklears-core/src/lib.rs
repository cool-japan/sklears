#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! # sklears-core - Core Traits and Utilities
//!
//! This crate provides the foundational traits, types, and utilities that power
//! the entire sklears machine learning ecosystem.
//!
//! ## Overview
//!
//! `sklears-core` defines the essential building blocks for machine learning in Rust:
//!
//! - **Core Traits**: `Estimator`, `Fit`, `Predict`, `Transform`, `Score`
//! - **Type System**: Type-safe state machines (Untrained/Trained)
//! - **Error Handling**: Comprehensive error types with context
//! - **Validation**: Input validation and consistency checks
//! - **Utilities**: Common helper functions and types
//! - **Parallel Processing**: Abstractions for parallel algorithms
//! - **Dataset Handling**: Data loading, splitting, and manipulation
//!
//! ## Core Traits
//!
//! ### Estimator
//!
//! The base trait for all machine learning models:
//!
//! ```rust,ignore
//! pub trait Estimator {
//!     type Config;
//!     type Error;
//! }
//! ```
//!
//! ### Fit
//!
//! Training an estimator on data:
//!
//! ```rust,ignore
//! pub trait Fit<X, Y> {
//!     type Fitted;
//!     fn fit(self, x: &X, y: &Y) -> Result<Self::Fitted, Self::Error>;
//! }
//! ```
//!
//! ### Predict
//!
//! Making predictions with a trained model:
//!
//! ```rust,ignore
//! pub trait Predict<X, Y> {
//!     fn predict(&self, x: &X) -> Result<Y, Self::Error>;
//! }
//! ```
//!
//! ### Transform
//!
//! Transforming data (for preprocessing and dimensionality reduction):
//!
//! ```rust,ignore
//! pub trait Transform<X> {
//!     fn transform(&self, x: &X) -> Result<X, Self::Error>;
//! }
//! ```
//!
//! ## Type-Safe State Machines
//!
//! Models use phantom types to track training state at compile time:
//!
//! ```rust,ignore
//! pub struct Untrained;
//! pub struct Trained;
//!
//! pub struct Model<State = Untrained> {
//!     config: ModelConfig,
//!     state: PhantomData<State>,
//!     weights: Option<Weights>, // Only Some in Trained state
//! }
//! ```
//!
//! This ensures:
//! - ✅ Can't predict with an untrained model (compile error)
//! - ✅ Can't accidentally re-train a trained model
//! - ✅ Type system enforces correct usage patterns
//!
//! ## Error Handling
//!
//! Comprehensive error types with rich context:
//!
//! ```rust,ignore
//! pub enum SklearsError {
//!     InvalidInput(String),
//!     ShapeMismatch { expected: Shape, got: Shape },
//!     NotFitted,
//!     ConvergenceError { iterations: usize },
//!     // ... and many more
//! }
//! ```
//!
//! ## Validation
//!
//! Input validation utilities ensure data consistency:
//!
//! ```rust,ignore
//! use sklears_core::validation;
//!
//! // Check that X and y have compatible shapes
//! validation::check_consistent_length(x, y)?;
//!
//! // Check for NaN/Inf values
//! validation::check_array(x)?;
//!
//! // Validate classification targets
//! validation::check_classification_targets(y)?;
//! ```
//!
//! ## Parallel Processing
//!
//! Abstractions for parallel algorithm execution:
//!
//! ```rust,ignore
//! use sklears_core::parallel::ParallelConfig;
//! use rayon::prelude::*;
//!
//! let config = ParallelConfig::new().n_jobs(-1); // Use all cores
//!
//! data.par_iter()
//!     .map(|sample| process(sample))
//!     .collect()
//! ```
//!
//! ## Feature Flags
//!
//! - `simd` - Enable SIMD optimizations
//! - `gpu_support` - GPU acceleration support
//! - `arrow` - Apache Arrow interoperability
//! - `binary` - Binary serialization support
//!
//! ## Examples
//!
//! See individual module documentation for detailed examples.
//!
//! ## Integration
//!
//! This crate is re-exported by the main `sklears` crate, so you typically don't
//! need to depend on it directly unless you're building custom estimators.

pub mod dataset;
pub mod distributed;
pub mod distributed_algorithms;
pub mod error;
pub mod parallel;
pub mod traits;
pub mod types;
pub mod utils;
pub mod validation;
pub mod validation_examples;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "gpu_support")]
pub mod gpu;

#[cfg(feature = "arrow")]
pub mod arrow;

#[cfg(feature = "binary")]
pub mod binary;

pub mod advanced_array_ops;
pub mod advanced_benchmarking;
pub mod algorithm_markers;
pub mod async_traits;
pub mod auto_benchmark_generation;
pub mod autodiff;
pub mod benchmarking;
pub mod compatibility;
pub mod compile_time_macros;
pub mod compile_time_validation;
// TODO: Complex generic testing - needs blanket trait implementations
// pub mod contract_testing;
pub mod contribution;
pub mod dependent_types;
pub mod derive_macros;
pub mod dsl_impl;
pub mod effect_types;
pub mod ensemble_improvements;
pub mod exhaustive_error_handling;
pub mod exotic_hardware;
pub mod exotic_hardware_impls;
pub mod fallback_strategies;
pub mod features;
pub mod formal_verification;
pub mod format_io;
pub mod formatting;
pub mod memory_safety;
pub mod mock_objects;
pub mod performance_profiling;
pub mod performance_reporting;
pub mod plugin;
pub mod plugin_marketplace_impl;
pub mod refinement_types;
pub mod streaming_lifetimes;
pub mod unsafe_audit;

// Export the procedural macros for DSL support
pub mod macros;

// Modularized API reference system (refactored from api_reference_generator.rs)
pub mod api_analyzers;
pub mod api_data_structures;
pub mod api_formatters;
pub mod api_generator_config;
pub mod interactive_api_reference;
pub mod interactive_playground;
pub mod search_engines;
pub mod tutorial_examples;
pub mod tutorial_system;
pub mod wasm_playground_impl;

// Trait explorer tool for interactive API navigation
pub mod trait_explorer;

// Public/private API boundaries
mod private;
pub mod public;

// Custom lints for ML-specific patterns
#[cfg(feature = "custom_lints")]
pub mod lints;

// Dependency audit and optimization
pub mod dependency_audit;

// Code coverage reporting and enforcement
pub mod code_coverage;

// Input sanitization for untrusted data
pub mod input_sanitization;

// TODO: HRTB issues - needs trait bound simplification
// #[allow(non_snake_case)]
// #[cfg(test)]
// pub mod property_tests;

// TODO: HRTB issues - needs trait bound simplification
// #[allow(non_snake_case)]
// #[cfg(test)]
// pub mod test_utilities;

pub mod prelude {
    /// Convenient re-exports of the most commonly used types and traits
    ///
    /// This prelude is organized by stability guarantees:
    /// - Stable APIs are always available
    /// - Experimental APIs require explicit opt-in
    /// - Deprecated APIs emit warnings
    // === Stable Public APIs (Always Available) ===
    // Core traits - guaranteed stable
    pub use crate::public::stable::{
        Estimator, Fit, FitPredict, FitTransform, PartialFit, Predict, Transform,
    };

    // Core types - guaranteed stable
    pub use crate::public::stable::{
        Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, FeatureCount,
        Features, Float, FloatBounds, Int, IntBounds, Labels, Numeric, Predictions, Probabilities,
        Probability, SampleCount, Target,
    };

    // Error handling - guaranteed stable
    pub use crate::public::stable::{ErrorChain, ErrorContext, Result, SklearsError};

    // Validation - guaranteed stable
    pub use crate::public::stable::{Validate, ValidationContext, ValidationRule};

    // Dataset utilities - guaranteed stable
    pub use crate::public::stable::{load_iris, make_blobs, make_regression, Dataset};

    // === Experimental APIs (Require Opt-in) ===

    #[cfg(feature = "experimental")]
    pub use crate::public::experimental::*;

    // === Additional Stable Exports ===

    // Zero-copy utilities - stable
    pub use crate::types::zero_copy::{
        array_views, dataset_ops, ArrayPool, ZeroCopyArray, ZeroCopyDataset,
    };
    pub use crate::types::{
        CowDataset, CowFeatures, CowLabels, CowPredictions, CowProbabilities, CowSampleWeight,
        CowTarget, Distances, SampleWeight, Similarities, ZeroCopy, ZeroCopyFeatures,
        ZeroCopyTarget,
    };

    // Validation utilities - stable
    pub use crate::validation::{ml as validation_ml, ConfigValidation, ValidationRules};

    // Compile-time validation - stable
    pub use crate::compile_time_validation::{
        CompileTimeValidated, DimensionValidator, LinearRegressionConfig,
        LinearRegressionConfigBuilder, ParameterValidator, PositiveValidator, ProbabilityValidator,
        RangeValidator, SolverCompatibility, ValidatedConfig,
    };

    // Memory-mapped datasets - stable when available
    #[cfg(feature = "mmap")]
    pub use crate::dataset::MmapDataset;

    // Arrow integration - stable when available
    #[cfg(feature = "arrow")]
    pub use crate::arrow::{ArrowDataset, ColumnStats};

    // Binary format support - stable when available
    #[cfg(feature = "binary")]
    pub use crate::binary::{
        convenience, ArrayBinaryFormat, BinaryConfig, BinaryDeserialize, BinaryFileStorage,
        BinaryFormat, BinaryMetadata, BinarySerialize, BinarySerializer, CompressionType,
        StreamingBinaryReader, StreamingBinaryWriter,
    };

    // SIMD operations - experimental, requires feature flag
    #[cfg(feature = "simd")]
    pub use crate::simd::{SimdArrayOps, SimdOps};

    // GPU acceleration - experimental, requires feature flag and CUDA
    #[cfg(feature = "gpu_support")]
    pub use crate::gpu::{
        GpuArray, GpuContext, GpuDeviceProperties, GpuMatrixOps, GpuMemoryInfo, GpuUtils,
        MemoryTransferOpts, TransferStrategy,
    };

    // Parallel processing - stable
    pub use crate::parallel::{
        ParallelConfig, ParallelCrossValidation, ParallelCrossValidator, ParallelEnsemble,
        ParallelEnsembleOps, ParallelFit, ParallelMatrixOps, ParallelPredict, ParallelTransform,
    };

    // Async traits - experimental
    #[cfg(feature = "async_support")]
    pub use crate::async_traits::{
        AsyncConfig, AsyncCrossValidation, AsyncEnsemble, AsyncFitAdvanced,
        AsyncHyperparameterOptimization, AsyncModelPersistence, AsyncPartialFit,
        AsyncPredictAdvanced, AsyncTransformAdvanced, CancellationToken, ConfidenceInterval,
        ProgressInfo,
    };

    // Plugin system - experimental
    #[cfg(feature = "plugins")]
    pub use crate::plugin::{
        AlgorithmPlugin, ClusteringPlugin, LogLevel, Plugin, PluginCapability, PluginCategory,
        PluginConfig, PluginConfigBuilder, PluginFactory, PluginLoader, PluginMetadata,
        PluginParameter, PluginRegistry, RuntimeSettings, TransformerPlugin,
    };

    // API stability utilities
    pub use crate::public::{
        api_version_info, is_api_experimental, is_api_stable, ApiStability, ApiVersionInfo,
        ExperimentalApi, PublicApiConfig, PublicApiConfigBuilder, StableApi,
    };

    // Custom lints for ML-specific patterns
    #[cfg(feature = "custom_lints")]
    pub use crate::lints::{
        ApiUsageLint, ArrayPerformanceLint, DataValidationLint, LintCategory, LintConfig,
        LintRegistry, LintRule, LintSeverity, MemoryLeakLint, ModelValidationLint,
        NumericalStabilityLint,
    };

    // Dependency audit and optimization
    pub use crate::dependency_audit::{
        calculate_metrics, generate_dependency_graph, BinarySizeImpact, CompileTimeImpact,
        DependencyAudit, DependencyCategory, DependencyInfo, DependencyRecommendation,
        DependencyReport, RecommendationAction,
    };

    // Code coverage reporting and enforcement
    pub use crate::code_coverage::{
        CICoverageResult, CIDConfig, CoverageCI, CoverageCollector, CoverageConfig, CoverageReport,
        CoverageTool, QualityGatesResult, RecommendationPriority,
    };

    // Input sanitization for untrusted data
    pub use crate::input_sanitization::{
        is_ml_data_safe, sanitize_ml_data, InputSanitizer, SafetyIssue, SanitizationConfig,
        Sanitize,
    };

    // Advanced array operations for high-performance computing
    pub use crate::advanced_array_ops::{ArrayStats, MatrixOps, MemoryOps};

    // Re-export the error_context macro
    pub use crate::error_context;

    // Code quality and safety tools - stable
    pub use crate::formatting::{
        CodeFormatter, FormattingConfig, FormattingConfigBuilder, FormattingIssue,
        FormattingReport, IssueSeverity, MLFormattingRules,
    };

    pub use crate::unsafe_audit::{
        SafetyRecommendation, SafetySeverity, UnsafeAuditConfig, UnsafeAuditReport, UnsafeAuditor,
        UnsafeFinding, UnsafePattern, UnsafeType,
    };

    // Memory safety guarantees and utilities - stable
    pub use crate::memory_safety::{
        MemoryPoolStats, MemorySafety, MemorySafetyGuarantee, SafeArrayOps, SafeMemoryPool,
        SafePooledBuffer, SafePtr, SafeSharedModel, UnsafeValidationResult,
    };

    // Benchmarking utilities - stable
    pub use crate::benchmarking::{
        AccuracyComparison, AlgorithmBenchmark, AlgorithmType, AutomatedBenchmarkRunner,
        BenchmarkConfig, BenchmarkDataset, BenchmarkResults, BenchmarkRunResult, BenchmarkSuite,
        MemoryStatistics, TimingStatistics,
    };

    // Mock objects for testing - now enabled and working
    pub use crate::mock_objects::{
        MockBehavior, MockConfig, MockEnsemble, MockErrorType, MockEstimator, MockEstimatorBuilder,
        MockStateSnapshot, MockTransformConfig, MockTransformType, MockTransformer,
        MockTransformerBuilder, TrainedMockEstimator, VotingStrategy,
    };

    // Contract testing framework - temporarily disabled until ndarray 0.17 migration is complete
    // pub use crate::contract_testing::{
    //     ContractTestConfig, ContractTestResult, ContractTestSummary, ContractTester,
    //     PropertyTestStats, TestCase, TraitLaws,
    // };

    // Compatibility layers for popular ML libraries - stable
    pub use crate::compatibility::{
        numpy::NumpyArray,
        pandas::{DataFrame, DataValue},
        pytorch::{ndarray_to_pytorch_tensor, TensorMetadata},
        serialization::{CrossPlatformModel, ModelFormat, ModelSerialization},
        sklearn::{FittedScikitLearnModel, ParamValue, ScikitLearnModel, SklearnCompatible},
    };

    // Standard format readers and writers - stable
    pub use crate::format_io::{
        CsvOptions, DataFormat, FormatDetector, FormatOptions, FormatReader, FormatWriter,
        Hdf5Options, JsonOptions, NumpyOptions, ParquetOptions, StreamingReader,
    };

    // Contribution guidelines and review process - stable
    pub use crate::contribution::{
        AlgorithmicCriteria, ClippyLevel, CodeQualityCriteria, ContributionChecker,
        ContributionConfig, ContributionResult, ContributionWorkflow, DocumentationCriteria,
        GateResult, PerformanceCriteria, QualityGate, QualityGateType, ReviewCriteria,
        TestingCriteria, WorkflowStep,
    };

    // Automated performance reporting system - stable
    pub use crate::performance_reporting::{
        AlertConfig, AnalysisResult, AnalysisType, HealthStatus, OutputFormat, PerformanceAnalyzer,
        PerformanceReport, PerformanceReporter, RegressionThreshold, ReportConfig, TimeRange,
        TrendDirection,
    };

    // Modularized API reference system - stable
    pub use crate::api_analyzers::{
        CrossReferenceBuilder as ModularCrossReferenceBuilder, ExampleValidator,
        TraitAnalyzer as ModularTraitAnalyzer, TypeExtractor as ModularTypeExtractor,
    };
    pub use crate::api_data_structures::{
        ApiReference as ModularApiReference, CodeExample as ModularCodeExample,
        TraitInfo as ModularTraitInfo, TypeInfo as ModularTypeInfo,
    };
    pub use crate::api_formatters::{
        ApiReferenceGenerator as ModularApiReferenceGenerator, DocumentFormatter,
    };
    pub use crate::api_generator_config::{
        GeneratorConfig as ModularGeneratorConfig, OutputFormat as ModularOutputFormat,
        ValidationConfig,
    };
    pub use crate::interactive_playground::{
        LiveCodeRunner, UIComponentBuilder, WasmPlaygroundManager,
    };
    pub use crate::search_engines::{
        AutocompleteTrie, SearchQuery, SearchResult, SemanticSearchEngine,
    };
    pub use crate::tutorial_system::{
        LearningPath, ProgressTracker, Tutorial, TutorialBuilder, TutorialSystem,
    };

    // Trait explorer tool for interactive API navigation - stable
    pub use crate::trait_explorer::{
        CompilationImpact, DependencyAnalysis, DependencyAnalyzer, EdgeType, ExampleCategory,
        ExampleDifficulty, ExampleGenerator, ExplorerConfig, GraphExportFormat, MemoryFootprint,
        PerformanceAnalysis, RuntimeOverhead, SimilarTrait, TraitExplorationResult, TraitExplorer,
        TraitGraph, TraitGraphEdge, TraitGraphGenerator, TraitGraphMetadata, TraitGraphNode,
        TraitNodeType, TraitPerformanceAnalyzer, TraitRegistry, UsageExample,
    };

    // Exotic hardware support - experimental (TPU, FPGA, Quantum)
    #[cfg(feature = "exotic_hardware")]
    pub use crate::exotic_hardware::{
        ActivationType, ComputationGraph, ComputationMetadata, ComputationNode, ComputationResult,
        ExoticHardware, ExoticHardwareManager, FpgaDevice, FpgaVendor, HardwareCapabilities,
        HardwareCompiler, HardwareComputation, HardwareId, HardwareMemoryManager, HardwareStatus,
        HardwareType, MemoryHandle, MemoryStats, Operation, PerformanceEstimate, Precision,
        QuantumBackend, QuantumDevice, TensorSpec, TpuDevice, TpuVersion, ValidationReport,
    };

    // Effect type system - experimental (compile-time effect tracking)
    #[cfg(feature = "effect_types")]
    pub use crate::effect_types::{
        AsyncEffect, Capability, Combined, Effect, EffectAnalyzer, EffectBuilder, EffectMetadata,
        EffectType, Fallible, FallibleIOEffect, GPUMemoryEffect, IORandomEffect, Linear, Memory,
        MemoryIOEffect, Pure, Random, GPU, IO,
    };

    // Automatic differentiation - experimental (forward/reverse mode AD)
    #[cfg(feature = "autodiff")]
    pub use crate::autodiff::{
        ADMode, AutodiffConfig, ComputationNode as ADNode, Dual, SymbolicExpression, Variable,
        VariableId,
    };

    // Distributed computing support - experimental (cluster-aware ML) - TEMPORARILY DISABLED
    // #[cfg(feature = "distributed")]
    // pub use crate::distributed::{
    //     ClusterInfo, ClusterNode, DistributedCluster, DistributedDataset, DistributedEstimator,
    //     DistributedMessage, DistributedMetrics, DistributedOptimizer, DistributedTraining,
    //     FaultTolerance, GradientAggregation, MessagePassing, NodeId, ParameterServer,
    // };

    // Compile-time macros and verification - experimental (model verification) - TEMPORARILY DISABLED
    // #[cfg(feature = "compile_time_macros")]
    // pub use crate::compile_time_macros::{
    //     validate_performance, verify_dimensions, verify_model, BenchmarkConfig as CompileTimeBenchmarkConfig,
    //     CompileTimeVerifiable, ComplexityAnalysis, DimensionVerifiable, MathematicallyVerifiable,
    //     OptimizationSuggestion, PerformanceTargets, ScalingBehavior, VerificationConfig,
    //     VerificationEngine, VerificationResult,
    // };

    // Automatic benchmark generation - experimental (performance testing)
    #[cfg(feature = "auto_benchmarks")]
    pub use crate::auto_benchmark_generation::{
        generate_benchmarks_for_type, AutoBenchmarkConfig, BenchmarkExecutor, BenchmarkGenerator,
        BenchmarkResult, BenchmarkType, ComplexityClass, GeneratedBenchmark,
        PerformanceEstimate as AutoBenchmarkPerformanceEstimate, RegressionDetector,
        ScalingDimension,
    };

    // Advanced ensemble method improvements - now enabled and working
    pub use crate::ensemble_improvements::{
        AggregationMethod, BaseEstimator, BaseEstimatorConfig, BaseEstimatorType,
        DistributedConfig, DistributedEnsemble, EnsembleConfig, EnsembleType,
        LoadBalancingStrategy, NodeRole, ParallelConfig as EnsembleParallelConfig,
        ParallelEnsemble as AdvancedParallelEnsemble, SamplingStrategy, TrainedBaseModel,
        TrainedParallelEnsemble, TrainingState,
    };
}
