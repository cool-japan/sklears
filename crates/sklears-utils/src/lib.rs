#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_borrow)]
//! Utility functions and helpers for sklears
//!
//! This crate provides common utilities used across the sklears ecosystem,
//! including data validation, array manipulation, random number generation,
//! and dataset creation utilities.
//!
//! # Examples
//!
//! ```rust
//! use sklears_utils::validation::check_consistent_length;
//! use sklears_utils::random::set_random_state;
//! use scirs2_core::ndarray::array;
//!
//! // Validation
//! let x = array![1, 2, 3];
//! let y = array![4, 5, 6];
//! assert!(check_consistent_length(&[&x, &y]).is_ok());
//!
//! // Random state
//! set_random_state(42);
//! ```

pub mod api_integration;
// Temporarily commented out due to missing submodules
// pub mod architecture;
pub mod array_utils;
pub mod cloud_storage;
pub mod config;
pub mod cross_validation;
pub mod data_generation;
pub mod data_pipeline;
pub mod data_structures;
pub mod database;
pub mod debug;
pub mod distributed_computing;
pub mod ensemble;
pub mod environment;
pub mod error_handling;
pub mod external_integration;
pub mod feature_engineering;
pub mod file_io;
pub mod gpu_computing;
pub mod linear_algebra;
pub mod logging;
pub mod math_utils;
pub mod memory;
pub mod metrics;
pub mod multiclass;
pub mod optimization;
pub mod parallel;
pub mod performance;
pub mod performance_regression;
pub mod preprocessing;
pub mod probabilistic;
pub mod profile_guided_optimization;
pub mod r_integration;
pub mod random;
pub mod simd;
pub mod spatial;
pub mod statistical;
pub mod stats;
pub mod text_processing;
pub mod time_series;
pub mod type_safety;
pub mod validation;
pub mod visualization;

#[allow(non_snake_case)]
#[cfg(test)]
mod property_tests;

// Re-export specific functions to avoid conflicts
pub use api_integration::{
    ApiClient, ApiConfig, ApiError, ApiMetrics, ApiRequest, ApiResponse, ApiService,
    Authentication, HttpMethod, MLApiPatterns, MethodStats, MockApiClient, RequestBuilder,
};
// Temporarily commented out due to missing submodules
/*
pub use architecture::{
    AspectContext, AspectManager, BackoffStrategy, ChainValidationRule, ChainValidationType,
    ComparisonOperator, ComponentError, ComponentFactory, ComponentRegistry, ConfigurationBuilder,
    ConfigurationPreset, ErrorHandleResult, ErrorHandler, Event, EventBus, EventError,
    EventHandler, EventRecord, ExecutionStats, FeatureModule, FluentApiBuilder, FluentChainBuilder,
    FluentCondition, FluentConditionType, FluentError, FluentErrorHandling, FluentExecutionResult,
    FluentExecutionStats, FluentOperation, FluentOperationType, FluentRetryPolicy,
    FluentUtilityChain, Hook, HookConfig, HookContext, HookError, HookErrorHandling,
    HookExecutionStats, HookRegistry, HookResult, HookType, MiddlewareContext, MiddlewareError,
    MiddlewarePipeline, ModuleConfig, ModuleError, ModuleRegistry, PipelineHookManager, Plugin,
    PluginContext, PluginError, PluginExecution, PluginManager, PluginResult,
    PresetApplicationResult, PresetBuilder, PresetError, PresetRegistry, RetryCondition,
    ServiceLifecycle, ServiceLocator, ServiceMetadata, UtilityContext, UtilityError,
    UtilityFunction, UtilityHookManager, UtilityRegistry, UtilityResult, UtilityValue,
    ValidationError, ValidationRule, ValidationRuleType,
};
*/
pub use array_utils::{
    argmax,
    argmin,
    argsort,
    array_add_constant_inplace,
    array_apply_inplace,
    array_concatenate,
    array_cumsum,
    array_describe,
    array_max,
    array_mean,
    array_mean_f64,
    array_median,
    array_min,
    array_min_max,
    array_min_max_normalize,
    array_min_max_normalize_inplace,
    array_percentile,
    array_quantiles,
    array_resize,
    array_reverse,

    array_scale_inplace,
    array_split,
    array_standardize,
    // In-place operations
    array_standardize_inplace,
    array_std,
    // Statistical functions
    array_sum,
    array_unique_counts,
    array_var,
    array_variance_f64,
    boolean_indexing_1d,
    boolean_indexing_2d,
    broadcast_shape,
    // Core utilities
    check_array_1d,
    check_array_2d,
    column_or_1d,
    compatible_layout,

    compress_1d,

    concatenate_2d,
    create_mask,
    densify_threshold,

    efficient_copy,
    // Advanced indexing
    fancy_indexing_1d,
    fancy_indexing_2d,
    fast_dot_product_f32,
    fast_dot_product_f64,
    fast_sum_f32,

    fast_sum_f64,
    filter_array,
    flatten_2d,
    get_strides,
    is_broadcastable,
    // Memory operations
    is_contiguous,
    label_counts,
    make_contiguous,
    normalize_array,
    pad_2d,

    put_1d,
    // Shape operations
    reshape_1d_to_2d,
    safe_indexing,
    safe_indexing_2d,
    // Sparse operations
    safe_sparse_dot,
    safe_sparse_dot_f32,
    safe_sparse_dot_f64,
    simd_add_arrays_f32,
    // SIMD operations
    simd_add_arrays_f64,
    simd_multiply_arrays_f32,
    simd_multiply_arrays_f64,
    simd_scale_array_f32,
    simd_scale_array_f64,
    slice_with_step,
    sparse_add,
    sparse_diag,
    sparse_transpose,
    split_2d,
    stack_1d,
    take_1d,
    tile_2d,
    transpose,
    unique_labels,
    where_condition,
    ArrayStatistics,
};
pub use cloud_storage::{
    CloudProvider, CloudStorageClient, CloudStorageConfig, CloudStorageFactory, CloudStorageUtils,
    MockCloudStorageClient, ObjectMetadata, StorageMetrics, SyncMode, SyncResult,
};
pub use config::{
    ArgParser, Config, ConfigBuilder, ConfigSource, ConfigValidator, ConfigValue, HotReloadConfig,
};
pub use cross_validation::{
    CVSplit, GroupKFold, LeaveOneGroupOut, StratifiedKFold, TimeSeriesSplit,
};
pub use data_generation::*;
pub use data_pipeline::{
    DataPipeline, MLPipelineBuilder, PipelineContext, PipelineMetrics, PipelineMonitor,
    PipelineResult, PipelineStep, StepMetrics, TransformStep,
};
pub use data_structures::{
    AtomicCounter, BinarySearchTree, BlockMatrix, ConcurrentHashMap, ConcurrentQueue,
    ConcurrentRingBuffer, Graph, RingBuffer, TreeNode, TreeStatistics, Trie, TrieStatistics,
    WeightedGraph, WorkQueue,
};
pub use database::{
    Connection, DatabaseConfig, DatabaseError, DatabasePool, Query, QueryBuilder, QueryResult,
    ResultSet, Transaction,
};
pub use debug::{
    ArrayDebugger, DebugContext, DiagnosticTools, MemoryDebugger, PerformanceDebugger,
    TestDataGenerator, TimingStats,
};
pub use distributed_computing::{
    ClusterConfig, ClusterNode, ClusterStats, DistributedCluster, DistributedError, DistributedJob,
    FaultDetector, JobExecution, JobPriority, JobScheduler, JobStatus, JobType, LoadBalancer,
    LoadMetrics, NodeCapabilities, NodeStatus, ResourceRequirements, ResourceUsage,
    SchedulingStrategy,
};
pub use ensemble::{
    AggregationStrategy, BaggingPredictor, Bootstrap, OOBScoreEstimator, StackingHelper,
};
pub use environment::{
    CacheInfo, CpuInfo, EnvironmentInfo, FeatureChecker, HardwareDetector, MemoryInfo, OSInfo,
    PerformanceCharacteristics, RuntimeInfo,
};
pub use error_handling::{
    create_error, create_error_at, EnhancedError, ErrorAggregator, ErrorContext, ErrorRecovery,
    ErrorReporter, ErrorStatistics, ErrorSummary, RecoveryStrategy,
};
pub use external_integration::{
    ArrayTransfer, CFunctionSignature, CParameter, CType, FFIUtils, PyArrayBuffer, PythonInterop,
    PythonParameter, PythonValue, WasmBuildConfig, WasmOptimization, WasmParameter, WasmType,
    WasmUtils,
};
pub use feature_engineering::{
    BinningStrategy, FeatureBinner, InteractionFeatures, PolynomialFeatures,
};
pub use file_io::{
    CompressionUtils, EfficientFileReader, EfficientFileWriter, FormatConverter,
    SerializationUtils, StreamProcessor,
};
pub use gpu_computing::{
    ActivationFunction, GpuArrayOps, GpuDevice, GpuError, GpuKernelExecution, GpuKernelInfo,
    GpuMemoryAllocation, GpuProfiler, GpuUtils, KernelStats, MemoryStats, MemoryTransferStats,
};
pub use linear_algebra::{
    ConditionNumber, EigenDecomposition, MatrixDecomposition, MatrixNorms, MatrixRank, MatrixUtils,
    Pseudoinverse,
};
pub use logging::{
    flush_global_logger, get_global_logger, set_global_level, ConsoleOutput, DistributedLogger,
    FileOutput, JsonFormatter, LogAnalysis, LogAnalyzer, LogEntry, LogLevel, LogStats, Logger,
    LoggerConfig, OperationStats, PerformanceLogger, TextFormatter,
};
pub use math_utils::{
    constants, NumericalPrecision, OverflowDetection, RobustArrayOps, SpecialFunctions,
};
pub use memory::{
    AllocationStats, GcHelper, LeakDetector, MemoryAlignment, MemoryMappedFile, MemoryMonitor,
    MemoryPool, MemoryValidator, SafeBuffer, SafePtr, SafeVec, StackGuard, TrackingAllocator,
};
pub use metrics::{
    bhattacharyya_distance, braycurtis_distance, canberra_distance, chebyshev_distance,
    cosine_distance, cosine_distance_f32, cosine_similarity, cosine_similarity_f32,
    euclidean_distance, euclidean_distance_f32, hamming_distance, hamming_distance_normalized,
    hellinger_distance, jaccard_distance, jaccard_similarity, jensen_shannon_divergence,
    kl_divergence, mahalanobis_distance, manhattan_distance, manhattan_distance_f32,
    minkowski_distance, wasserstein_1d,
};
pub use multiclass::*;
pub use optimization::{
    ConstraintHandler, ConstraintViolation, ConvergenceCriteria, ConvergenceStatus,
    GradientComputer, GradientMethod, LineSearch, LineSearchMethod, OptimizationHistory,
};
pub use parallel::{ParallelIterator, ParallelReducer, ThreadPool, WorkStealingQueue};
pub use performance::{
    BaselineMetrics, Benchmark, BenchmarkResult, MemoryTracker, ProfileReport, ProfileResult,
    Profiler, RegressionDetector, RegressionResult, Timer, TimerSummary,
};
pub use performance_regression::{
    PerformanceRegressionTester, RegressionTestResult, RegressionThresholds,
};
pub use preprocessing::{DataCleaner, DataQualityAssessor, FeatureScaler, OutlierDetector};
pub use probabilistic::{
    BloomFilter, BloomFilterStats, CountMinSketch, CountMinSketchStats, HyperLogLog,
    HyperLogLogStats, LSHash, LSHashStats, MinHash, MinHashStats,
};
pub use profile_guided_optimization::{
    BranchProfile, BranchType, CacheStatistics, DependencyChain, FunctionProfile,
    ImplementationEffort, InstructionMix, LoopProfile, MemoryAccessPattern, MemoryAccessType,
    OptimizationApplication, OptimizationOpportunity, OptimizationRecommendation,
    OptimizationReport, OptimizationRule, OptimizationType, PerformanceProfile, PerformanceTargets,
    ProfileError, ProfileGuidedOptimizer, ProfileSummary, ProfilerConfig, RiskLevel, StridePattern,
    TriggerCondition,
};
pub use r_integration::{
    RDataFrame, RError, RIntegration, RMatrix, RPackageManager, RScriptBuilder,
    RStatisticalFunctions, RValue,
};
pub use random::{
    bootstrap_indices, get_rng, importance_sampling, k_fold_indices, random_indices,
    random_permutation, random_weights, reservoir_sampling, set_random_state, shuffle_indices,
    stratified_split_indices, train_test_split_indices, weighted_sampling_without_replacement,
    DistributionSampler, ThreadSafeRng,
};
pub use simd::{
    SimdCapabilities, SimdDistanceOps, SimdF32Ops, SimdF64Ops, SimdMatrixOps, SimdStatsOps,
};
pub use spatial::{
    geographic::{CoordinateSystem, GeoBounds, GeoPoint, GeoUtils, Hemisphere},
    KdTree, OctTree, Point, QuadTree, RTree, Rectangle, SpatialHash, SpatialHashStats,
};
pub use statistical::{
    ConfidenceInterval, ConfidenceIntervals, CorrelationAnalysis, DistributionFitting,
    StatisticalTests, TestResult,
};
pub use text_processing::{
    RegexUtils, StringSimilarity, TextAnalysis, TextNormalizer, TextParser, UnicodeUtils,
};
pub use time_series::{
    AggregationMethod, LagFeatureGenerator, SlidingWindow, TemporalAggregator, TemporalIndex,
    TimeSeries, TimeSeriesPoint, TimeZoneUtils, Timestamp, TrendDirection, WindowStats,
};
pub use type_safety::{
    DataState, ExactSize, Kilograms, MatrixMul, Measurement, Meters, MinSize, ModelState,
    NonNegative, Normalized, One, Pixels, Positive, Seconds, Three, Trained, Two, TypedArray,
    Untrained, Unvalidated, Validated, ValidatedArray, Zero, D1, D2, D3,
};
pub use validation::*;
pub use visualization::{
    AxisConfig, BoxPlotData, ChartData, Color, HeatmapData, HistogramData, LinePlotData,
    MLVisualizationUtils, PlotData, PlotLayout, PlotMargin, PlotSummary, PlotUtils, Point2D,
    ScatterPlotData,
};

/// Common error type for utils
#[derive(thiserror::Error, Debug, Clone)]
pub enum UtilsError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Empty input")]
    EmptyInput,
    #[error("Invalid random state: {0}")]
    InvalidRandomState(String),
    #[error("Insufficient data: need at least {min} samples, got {actual}")]
    InsufficientData { min: usize, actual: usize },
}

impl From<UtilsError> for sklears_core::error::SklearsError {
    fn from(err: UtilsError) -> Self {
        sklears_core::error::SklearsError::InvalidInput(err.to_string())
    }
}

impl From<serde_json::Error> for UtilsError {
    fn from(err: serde_json::Error) -> Self {
        UtilsError::InvalidParameter(format!("JSON serialization error: {err}"))
    }
}

/// Type alias for utils results
pub type UtilsResult<T> = std::result::Result<T, UtilsError>;
