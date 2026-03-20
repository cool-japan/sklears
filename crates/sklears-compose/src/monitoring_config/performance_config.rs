//! Performance Monitoring Configuration
//!
//! This module contains all configuration structures related to performance monitoring,
//! profiling, benchmarking, and anomaly detection. It provides comprehensive control
//! over performance analysis and optimization capabilities.

use std::collections::HashMap;
use std::time::Duration;

/// Performance monitoring configuration
///
/// Controls all aspects of performance monitoring including profiling,
/// benchmarking, anomaly detection, and performance metrics collection.
///
/// # Architecture
///
/// The performance monitoring system provides multiple analysis layers:
///
/// ```text
/// Performance Monitoring
/// ├── Profiling (CPU, Memory, Network)
/// ├── Benchmarking (Baseline, Regression, Load)
/// ├── Anomaly Detection (Statistical, ML-based)
/// ├── Performance Metrics (Latency, Throughput, Resources)
/// └── Optimization Insights (Bottlenecks, Recommendations)
/// ```
///
/// # Performance Metrics
///
/// The system tracks various performance dimensions:
/// - Execution latency and timing
/// - Resource utilization (CPU, memory, I/O)
/// - Throughput and processing rates
/// - Error rates and failure patterns
/// - Queue depths and backlog analysis
///
/// # Usage Examples
///
/// ## Production Performance Monitoring
/// ```rust
/// use sklears_compose::monitoring_config::PerformanceMonitoringConfig;
///
/// let config = PerformanceMonitoringConfig::production();
/// ```
///
/// ## Development Profiling
/// ```rust
/// let config = PerformanceMonitoringConfig::development_profiling();
/// ```
///
/// ## Load Testing Configuration
/// ```rust
/// let config = PerformanceMonitoringConfig::load_testing();
/// ```
#[derive(Debug, Clone)]
pub struct PerformanceMonitoringConfig {
    /// Enable performance monitoring
    ///
    /// Global switch for all performance monitoring functionality.
    /// When disabled, no performance data will be collected.
    pub enabled: bool,

    /// Performance metrics to track
    ///
    /// Defines which categories of performance metrics should be collected.
    /// This allows selective monitoring to focus on relevant performance aspects.
    pub metrics: Vec<PerformanceMetricType>,

    /// Profiling configuration
    ///
    /// Controls detailed profiling of CPU, memory, and other resources
    /// to identify performance bottlenecks and optimization opportunities.
    pub profiling: ProfilingConfig,

    /// Benchmarking configuration
    ///
    /// Defines benchmark tests and performance baselines to track
    /// performance changes over time and detect regressions.
    pub benchmarking: BenchmarkingConfig,

    /// Anomaly detection configuration
    ///
    /// Controls automated detection of performance anomalies and outliers
    /// to identify potential issues before they impact users.
    pub anomaly_detection: AnomalyDetectionConfig,

    /// Performance thresholds and alerting
    ///
    /// Defines performance thresholds that trigger alerts when exceeded.
    pub thresholds: PerformanceThresholds,

    /// Optimization analysis configuration
    ///
    /// Controls automated analysis to identify optimization opportunities
    /// and provide performance improvement recommendations.
    pub optimization: OptimizationAnalysisConfig,
}

/// Performance metric types
///
/// Defines the categories of performance metrics that can be collected
/// and analyzed. Each type focuses on different aspects of system performance.
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceMetricType {
    /// CPU utilization across cores and processes
    ///
    /// Tracks CPU usage patterns, core utilization distribution,
    /// and processing efficiency metrics.
    CpuUtilization,

    /// Memory usage and allocation patterns
    ///
    /// Monitors memory consumption, allocation patterns, garbage collection,
    /// and memory leak detection.
    MemoryUsage,

    /// Execution latency and timing
    ///
    /// Measures response times, processing delays, and timing distributions
    /// for various operations and workflows.
    ExecutionLatency,

    /// Throughput rates and processing capacity
    ///
    /// Tracks the rate of operations, data processing throughput,
    /// and system capacity utilization.
    Throughput,

    /// Resource contention and competition
    ///
    /// Monitors competition for shared resources like locks, queues,
    /// and I/O channels that can impact performance.
    ResourceContention,

    /// Queue wait times and backlog analysis
    ///
    /// Tracks queue depths, wait times, and processing backlogs
    /// that indicate system bottlenecks.
    QueueWaitTimes,

    /// Error rates and failure patterns
    ///
    /// Monitors error frequencies, failure patterns, and their
    /// impact on overall system performance.
    ErrorRates,

    /// I/O performance and disk operations
    ///
    /// Tracks file system operations, disk I/O rates, and storage
    /// performance characteristics.
    IoPerformance,

    /// Network performance and communication
    ///
    /// Monitors network latency, bandwidth usage, and communication
    /// patterns that affect distributed system performance.
    NetworkPerformance,

    /// Custom performance metric
    ///
    /// User-defined performance metric with specific measurement characteristics.
    Custom {
        /// Unique name for the metric
        name: String,
        /// Unit of measurement
        unit: String,
        /// Description of what is being measured
        description: String,
    },
}

/// Profiling configuration
///
/// Controls detailed profiling capabilities to capture fine-grained
/// performance data for analysis and optimization.
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable profiling
    ///
    /// Master switch for all profiling functionality.
    pub enabled: bool,

    /// Profiling mode selection
    ///
    /// Defines what type of profiling to perform and how detailed it should be.
    pub mode: ProfilingMode,

    /// Sampling rate for profiling
    ///
    /// Controls how frequently profiling samples are taken.
    /// Higher rates provide more detail but increase overhead.
    pub sampling_rate: f64,

    /// Duration for profiling sessions
    ///
    /// How long each profiling session should run.
    /// Longer sessions provide more comprehensive data.
    pub session_duration: Duration,

    /// Profiler output configuration
    ///
    /// Controls where and how profiling data is stored and formatted.
    pub output: ProfilerOutputConfig,

    /// Call stack depth for profiling
    ///
    /// Maximum depth of call stacks to capture in profiles.
    /// Deeper stacks provide more context but increase overhead.
    pub max_stack_depth: usize,

    /// Enable symbol resolution
    ///
    /// Whether to resolve function names and source locations in profiles.
    /// Provides more readable profiles but requires debug information.
    pub resolve_symbols: bool,
}

/// Profiling modes
///
/// Defines different types of profiling that can be performed,
/// each focusing on different aspects of system performance.
#[derive(Debug, Clone)]
pub enum ProfilingMode {
    /// CPU profiling
    ///
    /// Focuses on CPU usage patterns, function call frequencies,
    /// and execution time distributions.
    Cpu,

    /// Memory profiling
    ///
    /// Tracks memory allocations, deallocations, and usage patterns
    /// to identify memory leaks and optimization opportunities.
    Memory,

    /// Combined CPU and memory profiling
    ///
    /// Captures both CPU and memory information simultaneously
    /// for comprehensive performance analysis.
    Both,

    /// I/O profiling
    ///
    /// Focuses on file system and network I/O operations
    /// to identify I/O bottlenecks and optimization opportunities.
    Io,

    /// Lock contention profiling
    ///
    /// Monitors lock usage, contention, and blocking patterns
    /// in multi-threaded applications.
    LockContention,

    /// Custom profiling mode
    ///
    /// User-defined profiling configuration for specific use cases.
    Custom {
        name: String,
        parameters: HashMap<String, String>,
    },
}

/// Profiler output configuration
///
/// Controls how profiling data is formatted, stored, and made available
/// for analysis and visualization.
#[derive(Debug, Clone)]
pub struct ProfilerOutputConfig {
    /// Output format for profiling data
    ///
    /// Defines the format used to store and export profiling results.
    pub format: ProfilerOutputFormat,

    /// Output destination path or URL
    ///
    /// Where to store profiling data. Can be a file path, database connection,
    /// or remote service endpoint.
    pub destination: String,

    /// Include detailed call traces
    ///
    /// Whether to include full call stack information in the output.
    /// Provides more detail but increases output size.
    pub detailed_traces: bool,

    /// Maximum output size per session
    ///
    /// Limits the size of profiling output to prevent excessive disk usage.
    pub max_size: usize,

    /// Enable real-time streaming
    ///
    /// Whether to stream profiling data in real-time or batch it.
    pub real_time: bool,

    /// Compression settings
    ///
    /// Whether to compress profiling data to reduce storage requirements.
    pub compression: CompressionConfig,
}

/// Profiler output formats
///
/// Defines the available formats for storing and exchanging profiling data.
#[derive(Debug, Clone)]
pub enum ProfilerOutputFormat {
    /// JSON format for human readability and API integration
    Json,
    /// Binary format for efficiency and performance
    Binary,
    /// Protocol Buffers for cross-platform compatibility
    Protobuf,
    /// Flame graph format for visualization
    FlameGraph,
    /// pprof format for Google's profiling tools
    Pprof,
    /// Custom format with user-defined structure
    Custom {
        /// Format name
        name: String,
        /// Format specification
        spec: String,
    },
}

/// Compression configuration for profiling data
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9, higher = more compression)
    pub level: u8,
}

/// Compression algorithms for profiling data
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    /// GZIP compression
    Gzip,
    /// LZ4 compression (fast)
    Lz4,
    /// Zstandard compression (balanced)
    Zstd,
    /// BZIP2 compression (high ratio)
    Bzip2,
}

/// Benchmarking configuration
///
/// Controls automated benchmarking to establish performance baselines,
/// track performance changes, and detect performance regressions.
#[derive(Debug, Clone)]
pub struct BenchmarkingConfig {
    /// Enable benchmarking
    pub enabled: bool,

    /// Benchmark suites to run
    ///
    /// List of benchmark test suites that should be executed.
    pub suites: Vec<BenchmarkSuite>,

    /// Benchmarking schedule
    ///
    /// When and how often to run benchmark tests.
    pub schedule: BenchmarkSchedule,

    /// Performance baselines
    ///
    /// Established performance baselines for comparison.
    pub baselines: HashMap<String, PerformanceBaseline>,

    /// Regression detection configuration
    ///
    /// Settings for detecting performance regressions automatically.
    pub regression_detection: RegressionDetectionConfig,
}

/// Benchmark test suite definition
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    /// Suite name
    pub name: String,

    /// Suite description
    pub description: String,

    /// Benchmark tests in this suite
    pub tests: Vec<BenchmarkTest>,

    /// Suite execution configuration
    pub config: BenchmarkExecutionConfig,
}

/// Individual benchmark test definition
#[derive(Debug, Clone)]
pub struct BenchmarkTest {
    /// Test name
    pub name: String,

    /// Test description
    pub description: String,

    /// Test type and configuration
    pub test_type: BenchmarkTestType,

    /// Performance expectations
    pub expectations: PerformanceExpectations,
}

/// Types of benchmark tests
#[derive(Debug, Clone)]
pub enum BenchmarkTestType {
    /// Latency benchmark (response time)
    Latency {
        /// Operations to benchmark
        operations: Vec<String>,
        /// Number of iterations
        iterations: u32,
    },

    /// Throughput benchmark (operations per second)
    Throughput {
        /// Operations to benchmark
        operations: Vec<String>,
        /// Test duration
        duration: Duration,
    },

    /// Load test (stress testing)
    Load {
        /// Concurrent users/operations
        concurrency: u32,
        /// Ramp-up duration
        ramp_up: Duration,
        /// Steady-state duration
        duration: Duration,
    },

    /// Memory usage benchmark
    Memory {
        /// Operations that allocate memory
        operations: Vec<String>,
        /// Maximum memory limit
        memory_limit: usize,
    },

    /// Custom benchmark
    Custom {
        /// Benchmark implementation
        implementation: String,
        /// Configuration parameters
        parameters: HashMap<String, String>,
    },
}

/// Performance expectations for benchmark tests
#[derive(Debug, Clone)]
pub struct PerformanceExpectations {
    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,

    /// Minimum required throughput
    pub min_throughput: Option<f64>,

    /// Maximum memory usage
    pub max_memory: Option<usize>,

    /// Maximum error rate (0.0 to 1.0)
    pub max_error_rate: Option<f64>,

    /// Custom performance criteria
    pub custom_criteria: HashMap<String, f64>,
}

/// Benchmark execution configuration
#[derive(Debug, Clone)]
pub struct BenchmarkExecutionConfig {
    /// Warmup iterations before measurement
    pub warmup_iterations: u32,

    /// Measurement iterations
    pub measurement_iterations: u32,

    /// Statistical confidence level
    pub confidence_level: f64,

    /// Maximum execution time per test
    pub timeout: Duration,

    /// Environment isolation settings
    pub isolation: IsolationSettings,
}

/// Environment isolation settings for benchmarks
#[derive(Debug, Clone)]
pub struct IsolationSettings {
    /// CPU affinity settings
    pub cpu_affinity: Option<Vec<usize>>,

    /// Memory allocation limits
    pub memory_limit: Option<usize>,

    /// Process priority
    pub priority: Option<i32>,

    /// Environment variables
    pub environment: HashMap<String, String>,
}

/// Benchmark scheduling configuration
#[derive(Debug, Clone)]
pub struct BenchmarkSchedule {
    /// Scheduling mode
    pub mode: ScheduleMode,

    /// Time zone for scheduling
    pub timezone: String,

    /// Maximum concurrent benchmark runs
    pub max_concurrent: u32,
}

/// Benchmark scheduling modes
#[derive(Debug, Clone)]
pub enum ScheduleMode {
    /// Run on a fixed interval
    Interval(Duration),

    /// Run on a cron schedule
    Cron(String),

    /// Run on demand only
    OnDemand,

    /// Run when triggered by events
    EventTriggered {
        /// Events that trigger benchmarks
        triggers: Vec<String>
    },
}

/// Performance baseline definition
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Baseline name
    pub name: String,

    /// Performance metrics and their baseline values
    pub metrics: HashMap<String, BaselineMetric>,

    /// When this baseline was established
    pub timestamp: std::time::SystemTime,

    /// Version or commit associated with this baseline
    pub version: String,

    /// Confidence intervals for the baseline
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// Baseline metric definition
#[derive(Debug, Clone)]
pub struct BaselineMetric {
    /// Metric value
    pub value: f64,

    /// Unit of measurement
    pub unit: String,

    /// Standard deviation
    pub std_dev: f64,

    /// Number of samples used
    pub sample_count: u32,
}

/// Regression detection configuration
#[derive(Debug, Clone)]
pub struct RegressionDetectionConfig {
    /// Enable regression detection
    pub enabled: bool,

    /// Statistical threshold for regression detection
    pub threshold: f64,

    /// Minimum number of samples for comparison
    pub min_samples: u32,

    /// Detection algorithms to use
    pub algorithms: Vec<RegressionDetectionAlgorithm>,

    /// Action to take when regression is detected
    pub on_regression: RegressionAction,
}

/// Regression detection algorithms
#[derive(Debug, Clone)]
pub enum RegressionDetectionAlgorithm {
    TTest,

    MannWhitney,

    Changepoint,

    MovingAverage { window: u32 },

    /// Custom algorithm
    Custom { name: String },
}

/// Actions to take when performance regression is detected
#[derive(Debug, Clone)]
pub enum RegressionAction {
    /// Log the regression
    Log,

    /// Send an alert
    Alert,

    /// Stop further testing
    Stop,

    /// Custom action
    Custom { action: String },
}

/// Anomaly detection configuration
///
/// Controls automated detection of performance anomalies to identify
/// potential issues before they impact system performance.
#[derive(Debug, Clone)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enabled: bool,

    /// Detection algorithms to use
    pub algorithms: Vec<AnomalyDetectionAlgorithm>,

    /// Sensitivity level (0.0 to 1.0)
    ///
    /// Higher values detect more anomalies but may have more false positives.
    pub sensitivity: f64,

    /// Training period for learning normal behavior
    pub training_period: Duration,

    /// Detection window size
    pub detection_window: Duration,

    /// Action to take when anomaly is detected
    pub on_anomaly: AnomalyAction,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone)]
pub enum AnomalyDetectionAlgorithm {
    ZScore { threshold: f64 },

    /// Isolation Forest algorithm
    IsolationForest,

    /// One-Class SVM
    OneClassSvm,

    /// Moving average deviation
    MovingAverage { window: u32, threshold: f64 },

    /// Seasonal decomposition
    SeasonalDecomposition,

    /// Custom algorithm
    Custom { name: String, config: HashMap<String, f64> },
}

/// Actions to take when performance anomaly is detected
#[derive(Debug, Clone)]
pub enum AnomalyAction {
    /// Log the anomaly
    Log,

    /// Send an alert
    Alert { severity: String },

    /// Trigger additional monitoring
    EnhancedMonitoring { duration: Duration },

    /// Custom action
    Custom { action: String },
}

/// Performance thresholds configuration
///
/// Defines performance thresholds that trigger alerts or actions
/// when exceeded.
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// CPU utilization thresholds
    pub cpu: ThresholdConfig,

    /// Memory usage thresholds
    pub memory: ThresholdConfig,

    /// Latency thresholds
    pub latency: ThresholdConfig,

    /// Throughput thresholds
    pub throughput: ThresholdConfig,

    /// Error rate thresholds
    pub error_rate: ThresholdConfig,

    /// Custom thresholds
    pub custom: HashMap<String, ThresholdConfig>,
}

/// Individual threshold configuration
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    /// Warning threshold
    pub warning: f64,

    /// Critical threshold
    pub critical: f64,

    /// Unit of measurement
    pub unit: String,

    /// Duration threshold must be exceeded
    pub duration: Duration,
}

/// Optimization analysis configuration
///
/// Controls automated analysis to identify optimization opportunities
/// and provide performance improvement recommendations.
#[derive(Debug, Clone)]
pub struct OptimizationAnalysisConfig {
    /// Enable optimization analysis
    pub enabled: bool,

    /// Analysis algorithms to run
    pub algorithms: Vec<OptimizationAlgorithm>,

    /// Analysis frequency
    pub frequency: Duration,

    /// Minimum improvement threshold to report
    pub min_improvement: f64,

    /// Historical data window for analysis
    pub analysis_window: Duration,
}

/// Optimization analysis algorithms
#[derive(Debug, Clone)]
pub enum OptimizationAlgorithm {
    /// Bottleneck analysis
    BottleneckAnalysis,

    /// Resource utilization analysis
    ResourceAnalysis,

    /// Performance pattern analysis
    PatternAnalysis,

    /// Cost-benefit analysis
    CostBenefit,

    /// Custom optimization algorithm
    Custom { name: String },
}

impl PerformanceMonitoringConfig {
    /// Create configuration optimized for production monitoring
    pub fn production() -> Self {
        Self {
            enabled: true,
            metrics: vec![
                PerformanceMetricType::CpuUtilization,
                PerformanceMetricType::MemoryUsage,
                PerformanceMetricType::ExecutionLatency,
                PerformanceMetricType::Throughput,
                PerformanceMetricType::ErrorRates,
            ],
            profiling: ProfilingConfig {
                enabled: true,
                mode: ProfilingMode::Both,
                sampling_rate: 0.01, // 1% sampling
                session_duration: Duration::from_secs(300),
                output: ProfilerOutputConfig {
                    format: ProfilerOutputFormat::Pprof,
                    destination: "./data/profiles".to_string(),
                    detailed_traces: false,
                    max_size: 100_000_000, // 100MB
                    real_time: false,
                    compression: CompressionConfig {
                        enabled: true,
                        algorithm: CompressionAlgorithm::Gzip,
                        level: 6,
                    },
                },
                max_stack_depth: 32,
                resolve_symbols: true,
            },
            benchmarking: BenchmarkingConfig {
                enabled: true,
                suites: Vec::new(),
                schedule: BenchmarkSchedule {
                    mode: ScheduleMode::Cron("0 2 * * *".to_string()), // Daily at 2 AM
                    timezone: "UTC".to_string(),
                    max_concurrent: 2,
                },
                baselines: HashMap::new(),
                regression_detection: RegressionDetectionConfig {
                    enabled: true,
                    threshold: 0.1, // 10% degradation
                    min_samples: 10,
                    algorithms: vec![RegressionDetectionAlgorithm::TTest],
                    on_regression: RegressionAction::Alert,
                },
            },
            anomaly_detection: AnomalyDetectionConfig {
                enabled: true,
                algorithms: vec![
                    AnomalyDetectionAlgorithm::ZScore { threshold: 3.0 },
                    AnomalyDetectionAlgorithm::MovingAverage { window: 10, threshold: 2.0 },
                ],
                sensitivity: 0.8,
                training_period: Duration::from_secs(86400 * 7), // 1 week
                detection_window: Duration::from_secs(300),
                on_anomaly: AnomalyAction::Alert { severity: "warning".to_string() },
            },
            thresholds: PerformanceThresholds {
                cpu: ThresholdConfig {
                    warning: 70.0,
                    critical: 90.0,
                    unit: "percent".to_string(),
                    duration: Duration::from_secs(300),
                },
                memory: ThresholdConfig {
                    warning: 80.0,
                    critical: 95.0,
                    unit: "percent".to_string(),
                    duration: Duration::from_secs(300),
                },
                latency: ThresholdConfig {
                    warning: 1000.0,
                    critical: 5000.0,
                    unit: "milliseconds".to_string(),
                    duration: Duration::from_secs(60),
                },
                throughput: ThresholdConfig {
                    warning: 100.0,
                    critical: 50.0,
                    unit: "requests_per_second".to_string(),
                    duration: Duration::from_secs(300),
                },
                error_rate: ThresholdConfig {
                    warning: 0.01,
                    critical: 0.05,
                    unit: "ratio".to_string(),
                    duration: Duration::from_secs(60),
                },
                custom: HashMap::new(),
            },
            optimization: OptimizationAnalysisConfig {
                enabled: true,
                algorithms: vec![
                    OptimizationAlgorithm::BottleneckAnalysis,
                    OptimizationAlgorithm::ResourceAnalysis,
                ],
                frequency: Duration::from_secs(86400), // Daily
                min_improvement: 0.05, // 5% improvement
                analysis_window: Duration::from_secs(86400 * 7), // 1 week
            },
        }
    }

    /// Create configuration optimized for development profiling
    pub fn development_profiling() -> Self {
        Self {
            enabled: true,
            metrics: vec![
                PerformanceMetricType::CpuUtilization,
                PerformanceMetricType::MemoryUsage,
                PerformanceMetricType::ExecutionLatency,
            ],
            profiling: ProfilingConfig {
                enabled: true,
                mode: ProfilingMode::Both,
                sampling_rate: 0.1, // 10% sampling for development
                session_duration: Duration::from_secs(60),
                output: ProfilerOutputConfig {
                    format: ProfilerOutputFormat::Json,
                    destination: "./dev/profiles".to_string(),
                    detailed_traces: true,
                    max_size: 10_000_000, // 10MB
                    real_time: true,
                    compression: CompressionConfig {
                        enabled: false,
                        algorithm: CompressionAlgorithm::Gzip,
                        level: 1,
                    },
                },
                max_stack_depth: 64,
                resolve_symbols: true,
            },
            benchmarking: BenchmarkingConfig {
                enabled: false, // Typically disabled in development
                suites: Vec::new(),
                schedule: BenchmarkSchedule {
                    mode: ScheduleMode::OnDemand,
                    timezone: "UTC".to_string(),
                    max_concurrent: 1,
                },
                baselines: HashMap::new(),
                regression_detection: RegressionDetectionConfig {
                    enabled: false,
                    threshold: 0.2,
                    min_samples: 5,
                    algorithms: vec![RegressionDetectionAlgorithm::TTest],
                    on_regression: RegressionAction::Log,
                },
            },
            anomaly_detection: AnomalyDetectionConfig {
                enabled: false, // Typically disabled in development
                algorithms: vec![],
                sensitivity: 0.5,
                training_period: Duration::from_secs(3600),
                detection_window: Duration::from_secs(60),
                on_anomaly: AnomalyAction::Log,
            },
            thresholds: PerformanceThresholds {
                cpu: ThresholdConfig {
                    warning: 90.0,
                    critical: 99.0,
                    unit: "percent".to_string(),
                    duration: Duration::from_secs(60),
                },
                memory: ThresholdConfig {
                    warning: 90.0,
                    critical: 99.0,
                    unit: "percent".to_string(),
                    duration: Duration::from_secs(60),
                },
                latency: ThresholdConfig {
                    warning: 10000.0,
                    critical: 30000.0,
                    unit: "milliseconds".to_string(),
                    duration: Duration::from_secs(60),
                },
                throughput: ThresholdConfig {
                    warning: 10.0,
                    critical: 1.0,
                    unit: "requests_per_second".to_string(),
                    duration: Duration::from_secs(60),
                },
                error_rate: ThresholdConfig {
                    warning: 0.1,
                    critical: 0.5,
                    unit: "ratio".to_string(),
                    duration: Duration::from_secs(60),
                },
                custom: HashMap::new(),
            },
            optimization: OptimizationAnalysisConfig {
                enabled: true,
                algorithms: vec![OptimizationAlgorithm::BottleneckAnalysis],
                frequency: Duration::from_secs(3600), // Hourly
                min_improvement: 0.1, // 10% improvement
                analysis_window: Duration::from_secs(3600), // 1 hour
            },
        }
    }

    /// Validate the performance monitoring configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate profiling sampling rate
        if self.profiling.sampling_rate < 0.0 || self.profiling.sampling_rate > 1.0 {
            return Err("Profiling sampling rate must be between 0.0 and 1.0".to_string());
        }

        // Validate anomaly detection sensitivity
        if self.anomaly_detection.sensitivity < 0.0 || self.anomaly_detection.sensitivity > 1.0 {
            return Err("Anomaly detection sensitivity must be between 0.0 and 1.0".to_string());
        }

        // Validate thresholds
        if self.thresholds.cpu.warning >= self.thresholds.cpu.critical {
            return Err("CPU warning threshold must be less than critical threshold".to_string());
        }

        if self.thresholds.memory.warning >= self.thresholds.memory.critical {
            return Err("Memory warning threshold must be less than critical threshold".to_string());
        }

        Ok(())
    }
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: vec![
                PerformanceMetricType::CpuUtilization,
                PerformanceMetricType::MemoryUsage,
                PerformanceMetricType::ExecutionLatency,
                PerformanceMetricType::Throughput,
            ],
            profiling: ProfilingConfig {
                enabled: false,
                mode: ProfilingMode::Cpu,
                sampling_rate: 0.01,
                session_duration: Duration::from_secs(60),
                output: ProfilerOutputConfig {
                    format: ProfilerOutputFormat::Json,
                    destination: "./data/profiles".to_string(),
                    detailed_traces: false,
                    max_size: 10_000_000,
                    real_time: false,
                    compression: CompressionConfig {
                        enabled: false,
                        algorithm: CompressionAlgorithm::Gzip,
                        level: 6,
                    },
                },
                max_stack_depth: 32,
                resolve_symbols: false,
            },
            benchmarking: BenchmarkingConfig {
                enabled: false,
                suites: Vec::new(),
                schedule: BenchmarkSchedule {
                    mode: ScheduleMode::OnDemand,
                    timezone: "UTC".to_string(),
                    max_concurrent: 1,
                },
                baselines: HashMap::new(),
                regression_detection: RegressionDetectionConfig {
                    enabled: false,
                    threshold: 0.1,
                    min_samples: 10,
                    algorithms: vec![],
                    on_regression: RegressionAction::Log,
                },
            },
            anomaly_detection: AnomalyDetectionConfig {
                enabled: false,
                algorithms: vec![],
                sensitivity: 0.5,
                training_period: Duration::from_secs(86400),
                detection_window: Duration::from_secs(300),
                on_anomaly: AnomalyAction::Log,
            },
            thresholds: PerformanceThresholds {
                cpu: ThresholdConfig {
                    warning: 80.0,
                    critical: 95.0,
                    unit: "percent".to_string(),
                    duration: Duration::from_secs(300),
                },
                memory: ThresholdConfig {
                    warning: 85.0,
                    critical: 95.0,
                    unit: "percent".to_string(),
                    duration: Duration::from_secs(300),
                },
                latency: ThresholdConfig {
                    warning: 1000.0,
                    critical: 5000.0,
                    unit: "milliseconds".to_string(),
                    duration: Duration::from_secs(60),
                },
                throughput: ThresholdConfig {
                    warning: 100.0,
                    critical: 50.0,
                    unit: "requests_per_second".to_string(),
                    duration: Duration::from_secs(300),
                },
                error_rate: ThresholdConfig {
                    warning: 0.05,
                    critical: 0.1,
                    unit: "ratio".to_string(),
                    duration: Duration::from_secs(300),
                },
                custom: HashMap::new(),
            },
            optimization: OptimizationAnalysisConfig {
                enabled: false,
                algorithms: vec![],
                frequency: Duration::from_secs(86400),
                min_improvement: 0.05,
                analysis_window: Duration::from_secs(86400),
            },
        }
    }
}