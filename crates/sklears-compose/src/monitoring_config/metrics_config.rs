//! Metrics Configuration
//!
//! This module contains all configuration structures related to metrics collection,
//! aggregation, storage, and custom metrics definitions. It provides comprehensive
//! control over how metrics are gathered and processed in the monitoring system.

use std::collections::HashMap;
use std::time::Duration;

/// Metrics collection configuration
///
/// Controls all aspects of metrics collection including what metrics to collect,
/// how often to collect them, and how to process and store the collected data.
///
/// # Architecture
///
/// The metrics system is designed with a flexible, extensible architecture:
///
/// ```text
/// MetricsConfig
/// ├── Collection Control (enabled, interval)
/// ├── Metric Types (system, application, custom)
/// ├── Aggregation (windowing, functions)
/// ├── Storage (backends, retention)
/// └── Custom Metrics (user-defined metrics)
/// ```
///
/// # Usage Examples
///
/// ## Basic Metrics Configuration
/// ```rust
/// use sklears_compose::monitoring_config::MetricsConfig;
///
/// let config = MetricsConfig::default();
/// ```
///
/// ## High-Performance Configuration
/// ```rust
/// let config = MetricsConfig::high_performance();
/// ```
///
/// ## Custom Metrics Setup
/// ```rust
/// let mut config = MetricsConfig::default();
/// config.add_custom_metric(CustomMetricDefinition {
///     name: "api_requests".to_string(),
///     description: "API request count".to_string(),
///     metric_type: CustomMetricType::Counter,
///     labels: vec!["endpoint".to_string(), "method".to_string()],
/// });
/// ```
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable metrics collection
    ///
    /// When disabled, no metrics will be collected, providing a way to
    /// completely turn off metrics collection for performance or privacy reasons.
    pub enabled: bool,

    /// Metrics collection interval
    ///
    /// Defines how frequently metrics are collected. Shorter intervals provide
    /// more granular data but increase system overhead.
    pub collection_interval: Duration,

    /// Types of metrics to collect
    ///
    /// Allows selective enabling of different metric categories to control
    /// the scope of metrics collection and system overhead.
    pub metric_types: Vec<MetricType>,

    /// Metrics aggregation configuration
    ///
    /// Controls how raw metric data is aggregated over time windows
    /// to provide summary statistics and reduce data volume.
    pub aggregation: MetricsAggregationConfig,

    /// Metrics storage configuration
    ///
    /// Defines where and how metrics data is stored, including
    /// storage backends, retention policies, and performance tuning.
    pub storage: MetricsStorageConfig,

    /// Custom metrics definitions
    ///
    /// User-defined metrics that extend the standard system metrics
    /// with application-specific measurements.
    pub custom_metrics: Vec<CustomMetricDefinition>,
}

/// Types of metrics to collect
///
/// Provides fine-grained control over which categories of metrics are collected.
/// This allows optimization of performance and storage by collecting only
/// the metrics that are needed for specific use cases.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    ExecutionTime,

    ResourceUtilization,

    Throughput,

    ErrorRates,

    QueueDepth,

    Latency,

    MemoryUsage,

    CpuUtilization,

    IoMetrics,

    NetworkMetrics,

    Custom {
        name: String,
        description: String
    },
}

/// Metrics aggregation configuration
///
/// Controls how raw metric measurements are aggregated over time windows
/// to provide summary statistics, reduce data volume, and enable efficient
/// querying and analysis.
#[derive(Debug, Clone)]
pub struct MetricsAggregationConfig {
    /// Aggregation window size
    ///
    /// Defines the time period over which metrics are aggregated.
    /// Larger windows provide smoother trends but less granular data.
    pub window_size: Duration,

    /// Aggregation functions to apply
    ///
    /// Specifies which statistical functions to compute for each metric.
    /// Common functions include mean, sum, min, max, percentiles.
    pub functions: Vec<AggregationFunction>,

    /// Maximum number of aggregation windows to retain
    ///
    /// Controls memory usage by limiting how many aggregated windows
    /// are kept in memory for fast access.
    pub max_windows: usize,

    /// Enable real-time aggregation
    ///
    /// When enabled, aggregations are computed continuously as data arrives.
    /// When disabled, aggregations are computed in batch at window boundaries.
    pub real_time: bool,
}

/// Statistical aggregation functions
///
/// Defines the types of statistical summaries that can be computed
/// from raw metric data over aggregation windows.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationFunction {
    /// Arithmetic mean (average)
    Mean,
    /// Sum of all values
    Sum,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of measurements
    Count,
    /// Median value (50th percentile)
    Median,
    /// Specific percentile (0.0 to 100.0)
    Percentile(f64),
    /// Standard deviation
    StandardDeviation,
    /// Rate of change per second
    Rate,
}

/// Metrics storage configuration
///
/// Defines how metrics data is persisted, including storage backends,
/// performance tuning, and data organization strategies.
#[derive(Debug, Clone)]
pub struct MetricsStorageConfig {
    /// Storage backend type
    ///
    /// Specifies which storage system to use for persisting metrics data.
    /// Different backends offer different performance and scalability characteristics.
    pub backend: StorageBackend,

    /// Batch size for storage operations
    ///
    /// Controls how many metrics are grouped together for efficient storage.
    /// Larger batches reduce I/O overhead but increase memory usage.
    pub batch_size: usize,

    /// Maximum buffer size before forced flush
    ///
    /// Prevents unlimited memory growth by forcing data to storage
    /// when the buffer reaches this size.
    pub max_buffer_size: usize,

    /// Storage flush interval
    ///
    /// Defines how often buffered data is written to storage.
    /// More frequent flushes reduce data loss risk but increase I/O.
    pub flush_interval: Duration,

    /// Enable data compression
    ///
    /// Reduces storage space and I/O at the cost of additional CPU usage.
    /// Particularly beneficial for high-volume metrics.
    pub compression: bool,

    /// Storage path or connection string
    ///
    /// Specifies where to store metrics data. Format depends on the backend:
    /// - File backends: directory path
    /// - Database backends: connection string
    /// - Cloud backends: bucket/container name
    pub storage_path: String,
}

/// Storage backend types
///
/// Defines the available storage systems for metrics persistence.
/// Each backend has different characteristics for performance, scalability,
/// and operational requirements.
#[derive(Debug, Clone)]
pub enum StorageBackend {
    Memory,

    File {
        format: FileFormat
    },

    /// Time-series database
    ///
    /// Optimized for time-series data with efficient compression and querying.
    /// Ideal for production metrics with long retention periods.
    TimeSeries {
        /// Database connection configuration
        connection: String
    },

    /// Cloud storage service
    ///
    /// Provides scalable, managed storage with high availability.
    /// Good for distributed deployments and long-term archival.
    Cloud {
        /// Cloud provider and configuration
        provider: String,
        /// Bucket or container name
        bucket: String
    },
}

/// File storage formats
///
/// Defines the serialization formats available for file-based storage.
/// Different formats offer trade-offs between size, speed, and compatibility.
#[derive(Debug, Clone)]
pub enum FileFormat {
    /// JSON format (human-readable, larger size)
    Json,
    /// Binary format (compact, faster)
    Binary,
    /// Compressed binary format (smallest size)
    CompressedBinary,
}

/// Custom metric definition
///
/// Allows users to define application-specific metrics that are collected
/// alongside the standard system metrics. Custom metrics follow the same
/// aggregation and storage patterns as built-in metrics.
#[derive(Debug, Clone)]
pub struct CustomMetricDefinition {
    /// Unique metric name
    ///
    /// Used to identify the metric in queries and storage.
    /// Should follow naming conventions (e.g., lowercase with underscores).
    pub name: String,

    /// Human-readable description
    ///
    /// Explains what this metric measures and how it should be interpreted.
    pub description: String,

    /// Type of custom metric
    ///
    /// Defines the measurement characteristics and how the metric behaves.
    pub metric_type: CustomMetricType,

    /// Label names for this metric
    ///
    /// Defines dimensions that can be used to slice and filter the metric data.
    /// For example, an HTTP request metric might have "endpoint" and "method" labels.
    pub labels: Vec<String>,

    /// Unit of measurement
    ///
    /// Describes the units for this metric (e.g., "seconds", "bytes", "requests/sec").
    pub unit: Option<String>,

    /// Help text for tooling and dashboards
    ///
    /// Extended description for use in monitoring dashboards and documentation.
    pub help: Option<String>,
}

/// Types of custom metrics
///
/// Defines the behavioral characteristics of custom metrics, determining
/// how they are collected, aggregated, and interpreted.
#[derive(Debug, Clone, PartialEq)]
pub enum CustomMetricType {
    Counter,

    Gauge,

    Histogram {
        buckets: Vec<f64>
    },

    /// Summary metric (quantiles over a sliding time window)
    ///
    /// Provides quantiles (percentiles) of observed values.
    /// Used for measurements where quantiles are more important than histograms.
    Summary {
        /// Quantiles to track (0.0 to 1.0)
        quantiles: Vec<f64>
    },
}

impl MetricsConfig {
    /// Create configuration optimized for high-performance scenarios
    ///
    /// High-performance configuration includes:
    /// - Fast collection intervals for real-time monitoring
    /// - In-memory storage for minimal latency
    /// - Selective metric types to reduce overhead
    /// - Real-time aggregation
    /// - Large batch sizes for efficiency
    pub fn high_performance() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_millis(100),
            metric_types: vec![
                MetricType::ExecutionTime,
                MetricType::Throughput,
                MetricType::ErrorRates,
                MetricType::Latency,
            ],
            aggregation: MetricsAggregationConfig {
                window_size: Duration::from_secs(1),
                functions: vec![
                    AggregationFunction::Mean,
                    AggregationFunction::Max,
                    AggregationFunction::Percentile(95.0),
                ],
                max_windows: 3600, // 1 hour of 1-second windows
                real_time: true,
            },
            storage: MetricsStorageConfig {
                backend: StorageBackend::Memory,
                batch_size: 1000,
                max_buffer_size: 10000,
                flush_interval: Duration::from_millis(100),
                compression: false,
                storage_path: "memory://metrics".to_string(),
            },
            custom_metrics: Vec::new(),
        }
    }

    /// Create configuration optimized for comprehensive monitoring
    ///
    /// Comprehensive configuration includes:
    /// - All metric types enabled
    /// - Persistent storage with compression
    /// - Multiple aggregation functions
    /// - Longer retention periods
    /// - Detailed custom metrics
    pub fn comprehensive() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_secs(5),
            metric_types: vec![
                MetricType::ExecutionTime,
                MetricType::ResourceUtilization,
                MetricType::Throughput,
                MetricType::ErrorRates,
                MetricType::QueueDepth,
                MetricType::Latency,
                MetricType::MemoryUsage,
                MetricType::CpuUtilization,
                MetricType::IoMetrics,
                MetricType::NetworkMetrics,
            ],
            aggregation: MetricsAggregationConfig {
                window_size: Duration::from_secs(60),
                functions: vec![
                    AggregationFunction::Mean,
                    AggregationFunction::Min,
                    AggregationFunction::Max,
                    AggregationFunction::Percentile(50.0),
                    AggregationFunction::Percentile(95.0),
                    AggregationFunction::Percentile(99.0),
                    AggregationFunction::StandardDeviation,
                ],
                max_windows: 1440, // 24 hours of 1-minute windows
                real_time: false,
            },
            storage: MetricsStorageConfig {
                backend: StorageBackend::File {
                    format: FileFormat::CompressedBinary
                },
                batch_size: 100,
                max_buffer_size: 1000,
                flush_interval: Duration::from_secs(10),
                compression: true,
                storage_path: "./data/metrics".to_string(),
            },
            custom_metrics: Vec::new(),
        }
    }

    /// Add a custom metric definition
    ///
    /// Adds a new custom metric to be collected alongside standard metrics.
    /// The metric will be validated to ensure it doesn't conflict with existing metrics.
    pub fn add_custom_metric(&mut self, metric: CustomMetricDefinition) {
        // Check for duplicate metric names
        if !self.custom_metrics.iter().any(|m| m.name == metric.name) {
            self.custom_metrics.push(metric);
        }
    }

    /// Remove a custom metric by name
    ///
    /// Removes a previously added custom metric definition.
    pub fn remove_custom_metric(&mut self, name: &str) {
        self.custom_metrics.retain(|m| m.name != name);
    }

    /// Get all metric names (built-in + custom)
    ///
    /// Returns a list of all metric names that will be collected,
    /// including both built-in metrics and custom metrics.
    pub fn get_all_metric_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // Add built-in metric names
        for metric_type in &self.metric_types {
            match metric_type {
                MetricType::ExecutionTime => names.push("execution_time".to_string()),
                MetricType::ResourceUtilization => names.push("resource_utilization".to_string()),
                MetricType::Throughput => names.push("throughput".to_string()),
                MetricType::ErrorRates => names.push("error_rates".to_string()),
                MetricType::QueueDepth => names.push("queue_depth".to_string()),
                MetricType::Latency => names.push("latency".to_string()),
                MetricType::MemoryUsage => names.push("memory_usage".to_string()),
                MetricType::CpuUtilization => names.push("cpu_utilization".to_string()),
                MetricType::IoMetrics => names.push("io_metrics".to_string()),
                MetricType::NetworkMetrics => names.push("network_metrics".to_string()),
                MetricType::Custom { name, .. } => names.push(name.clone()),
            }
        }

        // Add custom metric names
        for custom_metric in &self.custom_metrics {
            names.push(custom_metric.name.clone());
        }

        names
    }

    /// Validate the metrics configuration
    ///
    /// Checks for configuration issues that could prevent proper metrics collection.
    pub fn validate(&self) -> Result<(), String> {
        // Validate collection interval
        if self.collection_interval < Duration::from_millis(10) {
            return Err("Collection interval must be at least 10ms".to_string());
        }

        // Validate aggregation window size
        if self.aggregation.window_size < self.collection_interval {
            return Err("Aggregation window must be at least as large as collection interval".to_string());
        }

        // Validate custom metric names are unique
        let mut names = std::collections::HashSet::new();
        for metric in &self.custom_metrics {
            if !names.insert(&metric.name) {
                return Err(format!("Duplicate custom metric name: {}", metric.name));
            }
        }

        // Validate storage configuration
        if self.storage.batch_size == 0 {
            return Err("Storage batch size must be positive".to_string());
        }

        if self.storage.max_buffer_size < self.storage.batch_size {
            return Err("Max buffer size must be at least as large as batch size".to_string());
        }

        Ok(())
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_secs(10),
            metric_types: vec![
                MetricType::ExecutionTime,
                MetricType::ResourceUtilization,
                MetricType::ErrorRates,
                MetricType::Latency,
            ],
            aggregation: MetricsAggregationConfig::default(),
            storage: MetricsStorageConfig::default(),
            custom_metrics: Vec::new(),
        }
    }
}

impl Default for MetricsAggregationConfig {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(60),
            functions: vec![
                AggregationFunction::Mean,
                AggregationFunction::Max,
                AggregationFunction::Percentile(95.0),
            ],
            max_windows: 60, // 1 hour of 1-minute windows
            real_time: false,
        }
    }
}

impl Default for MetricsStorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::File {
                format: FileFormat::Json
            },
            batch_size: 100,
            max_buffer_size: 1000,
            flush_interval: Duration::from_secs(30),
            compression: false,
            storage_path: "./data/metrics".to_string(),
        }
    }
}