//! Configuration Types and Basic Structures
//!
//! This module contains all the configuration structures, enums, and basic types
//! used throughout the comprehensive benchmarking suite.

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

// ================================================================================================
// MAIN CONFIGURATION STRUCTURES
// ================================================================================================

/// Configuration for benchmarking suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingConfig {
    pub benchmark_categories: Vec<BenchmarkCategory>,
    pub performance_metrics: Vec<PerformanceMetric>,
    pub comparison_baselines: Vec<BaselineConfig>,
    pub regression_thresholds: RegressionThresholds,
    pub reporting_config: ReportingConfig,
    pub execution_config: ExecutionConfig,
}

/// Execution configuration for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub max_execution_time: Duration,
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub cooldown_period: Duration,
    pub parallel_execution: bool,
    pub resource_isolation: bool,
    pub environment_control: EnvironmentControl,
}

/// Environment control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentControl {
    pub cpu_affinity: Option<Vec<u32>>,
    pub memory_limit: Option<u64>,
    pub priority_class: Option<ProcessPriority>,
    pub environment_variables: HashMap<String, String>,
    pub working_directory: Option<String>,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    pub output_formats: Vec<OutputFormat>,
    pub visualization_types: Vec<VisualizationType>,
    pub detail_level: DetailLevel,
    pub include_charts: bool,
    pub include_raw_data: bool,
    pub export_paths: Vec<String>,
}

/// Thresholds for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionThresholds {
    pub performance_degradation: f64,
    pub accuracy_degradation: f64,
    pub memory_increase: f64,
    pub latency_increase: f64,
    pub statistical_significance: f64,
    pub consecutive_failures: u32,
}

/// Resource constraints for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_cpu_cores: Option<u32>,
    pub max_memory: Option<u64>,
    pub max_gpu_count: Option<u32>,
    pub max_concurrent_executions: u32,
    pub isolation_requirements: IsolationRequirements,
}

/// Isolation requirements for resource management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationRequirements {
    pub cpu_isolation: bool,
    pub memory_isolation: bool,
    pub network_isolation: bool,
    pub filesystem_isolation: bool,
    pub container_isolation: bool,
}

/// Resource requirements for benchmark tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub gpu_count: u32,
    pub disk_space_mb: u64,
    pub network_bandwidth_mbps: f64,
}

// ================================================================================================
// BENCHMARK CATEGORIES AND METRICS
// ================================================================================================

/// Benchmark categories for organization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    Performance,
    Scalability,
    Accuracy,
    MemoryEfficiency,
    Latency,
    Throughput,
    ResourceUtilization,
    StressTest,
    Custom(String),
}

/// Performance metrics for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub metric_name: String,
    pub metric_type: MetricType,
    pub unit: String,
    pub measurement_method: MeasurementMethod,
    pub aggregation_strategy: AggregationStrategy,
    pub quality_threshold: Option<QualityThreshold>,
}

/// Types of performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    ExecutionTime,
    MemoryUsage,
    CpuUtilization,
    GpuUtilization,
    NetworkIO,
    DiskIO,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    Throughput,
    Latency,
    Custom(String),
}

/// Methods for measuring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementMethod {
    Direct,
    Sampling,
    Statistical,
    Profiling,
    Instrumentation,
    Custom(String),
}

/// Strategies for aggregating measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStrategy {
    Mean,
    Median,
    Min,
    Max,
    Percentile(f64),
    StandardDeviation,
    GeometricMean,
    HarmonicMean,
    Custom(String),
}

/// Quality thresholds for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThreshold {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub target_value: Option<f64>,
    pub tolerance: f64,
    pub severity: ThresholdSeverity,
}

/// Severity levels for threshold violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

// ================================================================================================
// BASELINE AND COMPARISON CONFIGURATION
// ================================================================================================

/// Baseline configuration for comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfig {
    pub baseline_name: String,
    pub baseline_type: BaselineType,
    pub data_source: DataSource,
    pub update_policy: BaselineUpdatePolicy,
    pub validity_period: Option<Duration>,
}

/// Types of baselines for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineType {
    Historical,
    Theoretical,
    Competitive,
    UserDefined,
    Adaptive,
    Custom(String),
}

/// Data sources for baselines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    PreviousRuns,
    ExternalDatabase,
    ConfigurationFile,
    API(String),
    Manual,
    Custom(String),
}

/// Policies for updating baselines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineUpdatePolicy {
    Never,
    OnImprovement,
    Periodic(Duration),
    OnSignificantChange(f64),
    Adaptive,
    Custom(String),
}

// ================================================================================================
// EXECUTION AND SCHEDULING TYPES
// ================================================================================================

/// Process priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessPriority {
    Low,
    Normal,
    High,
    RealTime,
}

/// Scheduling policies for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    FIFO,
    Priority,
    ResourceBased,
    Adaptive,
    LoadBalanced,
    Custom(String),
}

/// Execution order for benchmark suites
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionOrder {
    Sequential,
    Parallel,
    Dependency,
    Priority,
    Custom(String),
}

/// Execution status tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout,
}

/// Types of execution errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    ParameterError,
    ResourceError,
    TimeoutError,
    ValidationError,
    SystemError,
    CustomError(String),
}

// ================================================================================================
// PARAMETER AND VALIDATION TYPES
// ================================================================================================

/// Parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Integer,
    Float,
    String,
    Boolean,
    Array(Box<ParameterType>),
    Enum(Vec<String>),
    Custom(String),
}

/// Parameter ranges for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    IntegerRange(i64, i64),
    FloatRange(f64, f64),
    StringLength(usize, usize),
    ArrayLength(usize, usize),
    Custom(String),
}

/// Types of validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    PreExecution,
    PostExecution,
    DuringExecution,
    MetricValidation,
    Custom(String),
}

/// Parameter definition for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    pub parameter_name: String,
    pub parameter_type: ParameterType,
    pub default_value: Option<String>,
    pub valid_range: Option<ParameterRange>,
    pub description: String,
    pub required: bool,
}

/// Validation rules for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_name: String,
    pub rule_type: ValidationRuleType,
    pub condition: String,
    pub error_message: String,
    pub severity: ThresholdSeverity,
}

/// Execution error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionError {
    pub error_type: ErrorType,
    pub error_message: String,
    pub stack_trace: Option<String>,
    pub recovery_suggestions: Vec<String>,
}

// ================================================================================================
// REPORTING AND VISUALIZATION TYPES
// ================================================================================================

/// Output formats for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    JSON,
    HTML,
    PDF,
    CSV,
    XML,
    Markdown,
    Custom(String),
}

/// Types of visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    LineChart,
    BarChart,
    ScatterPlot,
    Histogram,
    HeatMap,
    BoxPlot,
    ViolinPlot,
    Custom(String),
}

/// Detail levels for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetailLevel {
    Summary,
    Detailed,
    Comprehensive,
    Debug,
}

// ================================================================================================
// RESOURCE USAGE AND MONITORING TYPES
// ================================================================================================

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageInfo {
    pub peak_memory: u64,
    pub average_cpu: f64,
    pub peak_cpu: f64,
    pub gpu_utilization: Option<f64>,
    pub network_io: NetworkIOStats,
    pub disk_io: DiskIOStats,
    pub execution_time: Duration,
}

/// Network I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIOStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub connection_count: u32,
}

/// Disk I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOStats {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_operations: u64,
    pub write_operations: u64,
    pub average_seek_time: Duration,
}

// ================================================================================================
// STORAGE AND RETENTION TYPES
// ================================================================================================

/// Indexing strategies for result storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingStrategy {
    BTree,
    Hash,
    TimeSeriesIndex,
    CompositeIndex,
    Custom(String),
}

/// Retention policies for benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub max_age: Duration,
    pub max_count: Option<u32>,
    pub archive_policy: ArchivePolicy,
    pub compression_config: CompressionConfig,
}

/// Archive policies for old results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchivePolicy {
    Delete,
    Archive,
    Compress,
    ExternalStorage(String),
    Custom(String),
}

/// Compression configuration for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,
    pub compression_level: u32,
    pub enable_encryption: bool,
    pub chunk_size: usize,
}

/// Compression algorithms for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Brotli,
    Custom(String),
}

// ================================================================================================
// ANALYSIS AND FORECASTING TYPES
// ================================================================================================

/// Statistical analysis methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalMethod {
    TTest,
    WilcoxonRankSum,
    KruskalWallis,
    ANOVA,
    ChiSquare,
    FisherExact,
    Custom(String),
}

/// Trend analysis methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendAnalysisMethod {
    LinearRegression,
    PolynomialRegression,
    MovingAverage,
    ExponentialSmoothing,
    ARIMA,
    SeasonalDecomposition,
    Custom(String),
}

/// Anomaly detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionMethod {
    StatisticalOutlier,
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    DBSCAN,
    Custom(String),
}

/// Forecasting model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForecastingModelType {
    LinearRegression,
    ARIMA,
    ExponentialSmoothing,
    NeuralNetwork,
    EnsembleModel,
    Custom(String),
}

/// Confidence interval types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceIntervalType {
    Normal,
    Bootstrap,
    Bayesian,
    Custom(String),
}

// ================================================================================================
// COMPARISON AND RANKING TYPES
// ================================================================================================

/// Comparison methods for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonMethod {
    AbsoluteDifference,
    RelativeDifference,
    PercentageChange,
    StandardizedDifference,
    RankBased,
    StatisticalTest,
    Custom(String),
}

/// Ranking methods for results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RankingMethod {
    Simple,
    Weighted,
    Pareto,
    TOPSIS,
    ELECTRE,
    Custom(String),
}

/// Significance test types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignificanceTestType {
    TTest,
    WilcoxonRankSum,
    MannWhitneyU,
    KruskalWallis,
    FriedmanTest,
    Custom(String),
}

// ================================================================================================
// ERROR TYPES
// ================================================================================================

/// Benchmarking errors
#[derive(Debug, thiserror::Error)]
pub enum BenchmarkingError {
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Resource error: {0}")]
    ResourceError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("Comparison error: {0}")]
    ComparisonError(String),
    #[error("Reporting error: {0}")]
    ReportingError(String),
}

// ================================================================================================
// DEFAULT IMPLEMENTATIONS
// ================================================================================================

impl Default for BenchmarkingConfig {
    fn default() -> Self {
        Self {
            benchmark_categories: vec![BenchmarkCategory::Performance],
            performance_metrics: vec![PerformanceMetric::default()],
            comparison_baselines: vec![BaselineConfig::default()],
            regression_thresholds: RegressionThresholds::default(),
            reporting_config: ReportingConfig::default(),
            execution_config: ExecutionConfig::default(),
        }
    }
}

impl Default for PerformanceMetric {
    fn default() -> Self {
        Self {
            metric_name: "execution_time".to_string(),
            metric_type: MetricType::ExecutionTime,
            unit: "milliseconds".to_string(),
            measurement_method: MeasurementMethod::Direct,
            aggregation_strategy: AggregationStrategy::Mean,
            quality_threshold: None,
        }
    }
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            baseline_name: "historical".to_string(),
            baseline_type: BaselineType::Historical,
            data_source: DataSource::PreviousRuns,
            update_policy: BaselineUpdatePolicy::OnImprovement,
            validity_period: Some(Duration::from_secs(86400 * 30)), // 30 days
        }
    }
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            performance_degradation: 0.1, // 10% degradation
            accuracy_degradation: 0.05,   // 5% degradation
            memory_increase: 0.2,         // 20% increase
            latency_increase: 0.15,       // 15% increase
            statistical_significance: 0.05, // p < 0.05
            consecutive_failures: 3,
        }
    }
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            output_formats: vec![OutputFormat::JSON, OutputFormat::HTML],
            visualization_types: vec![VisualizationType::LineChart, VisualizationType::BarChart],
            detail_level: DetailLevel::Detailed,
            include_charts: true,
            include_raw_data: false,
            export_paths: vec!["./benchmark_reports".to_string()],
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(3600), // 1 hour
            warmup_iterations: 3,
            measurement_iterations: 10,
            cooldown_period: Duration::from_secs(5),
            parallel_execution: true,
            resource_isolation: false,
            environment_control: EnvironmentControl::default(),
        }
    }
}

impl Default for EnvironmentControl {
    fn default() -> Self {
        Self {
            cpu_affinity: None,
            memory_limit: None,
            priority_class: Some(ProcessPriority::Normal),
            environment_variables: HashMap::new(),
            working_directory: None,
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_cores: None,
            max_memory: None,
            max_gpu_count: None,
            max_concurrent_executions: 4,
            isolation_requirements: IsolationRequirements::default(),
        }
    }
}

impl Default for IsolationRequirements {
    fn default() -> Self {
        Self {
            cpu_isolation: false,
            memory_isolation: false,
            network_isolation: false,
            filesystem_isolation: false,
            container_isolation: false,
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_mb: 1024,
            gpu_count: 0,
            disk_space_mb: 1024,
            network_bandwidth_mbps: 100.0,
        }
    }
}

impl Default for ResourceUsageInfo {
    fn default() -> Self {
        Self {
            peak_memory: 0,
            average_cpu: 0.0,
            peak_cpu: 0.0,
            gpu_utilization: None,
            network_io: NetworkIOStats::default(),
            disk_io: DiskIOStats::default(),
            execution_time: Duration::from_secs(0),
        }
    }
}

impl Default for NetworkIOStats {
    fn default() -> Self {
        Self {
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            connection_count: 0,
        }
    }
}

impl Default for DiskIOStats {
    fn default() -> Self {
        Self {
            bytes_read: 0,
            bytes_written: 0,
            read_operations: 0,
            write_operations: 0,
            average_seek_time: Duration::from_nanos(0),
        }
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age: Duration::from_secs(86400 * 365), // 1 year
            max_count: Some(10000),
            archive_policy: ArchivePolicy::Compress,
            compression_config: CompressionConfig::default(),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Zstd,
            compression_level: 3,
            enable_encryption: false,
            chunk_size: 64 * 1024, // 64KB
        }
    }
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

/// Create a default performance metric for a specific type
pub fn create_performance_metric(
    name: &str,
    metric_type: MetricType,
    unit: &str,
) -> PerformanceMetric {
    PerformanceMetric {
        metric_name: name.to_string(),
        metric_type,
        unit: unit.to_string(),
        measurement_method: MeasurementMethod::Direct,
        aggregation_strategy: AggregationStrategy::Mean,
        quality_threshold: None,
    }
}

/// Create a quality threshold with target value
pub fn create_quality_threshold(
    target: f64,
    tolerance: f64,
    severity: ThresholdSeverity,
) -> QualityThreshold {
    QualityThreshold {
        min_value: Some(target - tolerance),
        max_value: Some(target + tolerance),
        target_value: Some(target),
        tolerance,
        severity,
    }
}

/// Create resource requirements for a benchmark
pub fn create_resource_requirements(
    cpu_cores: u32,
    memory_mb: u64,
    gpu_count: u32,
) -> ResourceRequirements {
    ResourceRequirements {
        cpu_cores,
        memory_mb,
        gpu_count,
        disk_space_mb: 1024,
        network_bandwidth_mbps: 100.0,
    }
}

/// Create a validation rule for parameter checking
pub fn create_validation_rule(
    name: &str,
    rule_type: ValidationRuleType,
    condition: &str,
    error_message: &str,
) -> ValidationRule {
    ValidationRule {
        rule_name: name.to_string(),
        rule_type,
        condition: condition.to_string(),
        error_message: error_message.to_string(),
        severity: ThresholdSeverity::Error,
    }
}

/// Create a baseline configuration for historical comparison
pub fn create_historical_baseline(name: &str) -> BaselineConfig {
    BaselineConfig {
        baseline_name: name.to_string(),
        baseline_type: BaselineType::Historical,
        data_source: DataSource::PreviousRuns,
        update_policy: BaselineUpdatePolicy::OnImprovement,
        validity_period: Some(Duration::from_secs(86400 * 30)),
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_benchmarking_config() {
        let config = BenchmarkingConfig::default();
        assert!(!config.benchmark_categories.is_empty());
        assert!(!config.performance_metrics.is_empty());
        assert_eq!(config.regression_thresholds.performance_degradation, 0.1);
    }

    #[test]
    fn test_performance_metric_creation() {
        let metric = create_performance_metric(
            "test_metric",
            MetricType::ExecutionTime,
            "ms"
        );
        assert_eq!(metric.metric_name, "test_metric");
        assert!(matches!(metric.metric_type, MetricType::ExecutionTime));
        assert_eq!(metric.unit, "ms");
    }

    #[test]
    fn test_quality_threshold_creation() {
        let threshold = create_quality_threshold(100.0, 5.0, ThresholdSeverity::Warning);
        assert_eq!(threshold.target_value, Some(100.0));
        assert_eq!(threshold.min_value, Some(95.0));
        assert_eq!(threshold.max_value, Some(105.0));
        assert!(matches!(threshold.severity, ThresholdSeverity::Warning));
    }

    #[test]
    fn test_resource_requirements_creation() {
        let requirements = create_resource_requirements(4, 8192, 1);
        assert_eq!(requirements.cpu_cores, 4);
        assert_eq!(requirements.memory_mb, 8192);
        assert_eq!(requirements.gpu_count, 1);
    }

    #[test]
    fn test_validation_rule_creation() {
        let rule = create_validation_rule(
            "test_rule",
            ValidationRuleType::PreExecution,
            "value > 0",
            "Value must be positive"
        );
        assert_eq!(rule.rule_name, "test_rule");
        assert!(matches!(rule.rule_type, ValidationRuleType::PreExecution));
        assert_eq!(rule.condition, "value > 0");
    }

    #[test]
    fn test_baseline_config_creation() {
        let baseline = create_historical_baseline("test_baseline");
        assert_eq!(baseline.baseline_name, "test_baseline");
        assert!(matches!(baseline.baseline_type, BaselineType::Historical));
        assert!(matches!(baseline.data_source, DataSource::PreviousRuns));
    }

    #[test]
    fn test_default_execution_config() {
        let config = ExecutionConfig::default();
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.measurement_iterations, 10);
        assert!(config.parallel_execution);
        assert!(!config.resource_isolation);
    }

    #[test]
    fn test_default_regression_thresholds() {
        let thresholds = RegressionThresholds::default();
        assert_eq!(thresholds.performance_degradation, 0.1);
        assert_eq!(thresholds.accuracy_degradation, 0.05);
        assert_eq!(thresholds.consecutive_failures, 3);
    }

    #[test]
    fn test_parameter_type_variants() {
        let types = vec![
            ParameterType::Integer,
            ParameterType::Float,
            ParameterType::String,
            ParameterType::Boolean,
            ParameterType::Array(Box::new(ParameterType::Integer)),
            ParameterType::Enum(vec!["a".to_string(), "b".to_string()]),
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_execution_status_variants() {
        let statuses = vec![
            ExecutionStatus::Queued,
            ExecutionStatus::Running,
            ExecutionStatus::Completed,
            ExecutionStatus::Failed,
            ExecutionStatus::Cancelled,
            ExecutionStatus::Timeout,
        ];
        assert_eq!(statuses.len(), 6);
    }
}