//! Core Metrics Data Types and Definitions
//!
//! This module provides the fundamental data types, error handling, and core
//! definitions for the distributed metrics collection system. It includes
//! metric value types, metadata structures, and basic collection interfaces.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Comprehensive error types for metrics collection operations
///
/// Provides detailed error categorization for different failure modes
/// in the metrics collection system with appropriate context.
#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Metric not found: {0}")]
    MetricNotFound(String),
    #[error("Invalid metric value: {0}")]
    InvalidMetricValue(String),
    #[error("Collection error: {0}")]
    CollectionError(String),
    #[error("Aggregation error: {0}")]
    AggregationError(String),
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("Export error: {0}")]
    ExportError(String),
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
    #[error("Scheduler error: {0}")]
    SchedulerError(String),
    #[error("Worker error: {0}")]
    WorkerError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for metrics operations
pub type MetricsResult<T> = Result<T, MetricsError>;

/// Comprehensive metric type classification
///
/// Defines all supported metric types for collection and analysis
/// with appropriate semantic meaning for each category.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Monotonically increasing counter
    Counter,
    /// Instantaneous measurement value
    Gauge,
    /// Distribution of values with buckets
    Histogram,
    /// Statistical summary with quantiles
    Summary,
    /// Duration measurement
    Timer,
    /// Rate of change over time
    Rate,
    /// Ratio between two values
    Ratio,
    /// Percentage value (0-100)
    Percentage,
    /// Statistical distribution
    Distribution,
    /// Unique value set
    Set,
    /// Custom application-specific metric
    Custom(String),
}

/// Flexible metric value representation
///
/// Supports multiple data types to accommodate different metric sources
/// and collection scenarios with proper serialization support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricValue {
    /// 64-bit signed integer
    Integer(i64),
    /// 64-bit floating point
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// String value
    String(String),
    /// Duration value
    Duration(Duration),
    /// Binary data
    Bytes(Vec<u8>),
    /// Array of metric values
    Array(Vec<MetricValue>),
    /// Structured object with key-value pairs
    Object(HashMap<String, MetricValue>),
}

/// Individual metric data point with timestamp and metadata
///
/// Represents a single measurement with associated contextual information
/// including tags, metadata, and data quality indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    /// Data point timestamp
    pub timestamp: SystemTime,
    /// Measured value
    pub value: MetricValue,
    /// Contextual tags for grouping and filtering
    pub tags: HashMap<String, String>,
    /// Comprehensive metadata
    pub metadata: MetricMetadata,
}

/// Comprehensive metadata for metric data points
///
/// Provides detailed context about data collection including quality,
/// collection method, processing information, and validation status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricMetadata {
    /// Data source identifier
    pub source: String,
    /// Data quality assessment
    pub quality: DataQuality,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Collection method used
    pub collection_method: CollectionMethod,
    /// Processing information
    pub processing_info: ProcessingInfo,
    /// Validation status
    pub validation_status: ValidationStatus,
}

/// Data quality indicators for metrics
///
/// Provides assessment of data reliability and accuracy
/// with specific degradation reasons when applicable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataQuality {
    /// High quality, reliable data
    High,
    /// Medium quality, generally reliable
    Medium,
    /// Low quality, use with caution
    Low,
    /// Quality unknown or unassessed
    Unknown,
    /// Degraded quality with specific reason
    Degraded(String),
}

/// Collection method classification
///
/// Identifies how the metric data was collected
/// for proper interpretation and processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionMethod {
    Push,
    Pull,
    Stream,
    Event,
    Scheduled,
    OnDemand,
    Batch,
    Custom(String),
}

/// Processing information for data points
///
/// Tracks processing operations applied to the data
/// for traceability and debugging purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingInfo {
    /// Processing pipeline identifier
    pub pipeline_id: String,
    /// Processing stages applied
    pub stages: Vec<String>,
    /// Processing timestamp
    pub processed_at: SystemTime,
    /// Processing duration
    pub processing_duration: Duration,
    /// Processing version
    pub processing_version: String,
}

/// Sampling information for data collection
///
/// Tracks sampling configuration and statistics
/// for understanding data representativeness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingInfo {
    /// Sampling method used
    pub method: SamplingMethod,
    /// Sampling rate (0.0 - 1.0)
    pub rate: f64,
    /// Sample size
    pub sample_size: u64,
    /// Total population size
    pub population_size: Option<u64>,
    /// Sampling interval
    pub interval: Option<Duration>,
}

/// Sampling method types
///
/// Defines different approaches to data sampling
/// for performance optimization and data reduction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// No sampling (collect all data)
    None,
    /// Random sampling
    Random,
    /// Systematic sampling (every nth item)
    Systematic,
    /// Stratified sampling
    Stratified,
    /// Reservoir sampling
    Reservoir,
    /// Adaptive sampling based on conditions
    Adaptive,
    /// Custom sampling method
    Custom(String),
}

/// Validation status for metric data
///
/// Indicates whether data has passed validation checks
/// and provides details about validation failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Data passed all validation checks
    Valid,
    /// Data failed validation with reasons
    Invalid(Vec<String>),
    /// Data partially valid with warnings
    Warning(Vec<String>),
    /// Data not yet validated
    Pending,
    /// Validation skipped
    Skipped,
    /// Validation error occurred
    Error(String),
}

/// Processing stage types for metric data
///
/// Defines different stages in the processing pipeline
/// for proper data transformation and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStageType {
    /// Data ingestion stage
    Ingestion,
    /// Data validation stage
    Validation,
    /// Data transformation stage
    Transformation,
    /// Data enrichment stage
    Enrichment,
    /// Data aggregation stage
    Aggregation,
    /// Data normalization stage
    Normalization,
    /// Data filtering stage
    Filtering,
    /// Data storage stage
    Storage,
    /// Custom processing stage
    Custom(String),
}

/// Error handling strategy for processing
///
/// Defines how errors should be handled during
/// data processing operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandlingStrategy {
    /// Stop processing on first error
    FailFast,
    /// Continue processing, collect errors
    ContinueOnError,
    /// Skip invalid data, continue processing
    SkipInvalid,
    /// Retry failed operations
    Retry { max_attempts: u32, delay: Duration },
    /// Custom error handling
    Custom(String),
}

/// Comparison operators for filtering and validation
///
/// Standard comparison operations used throughout
/// the metrics system for data filtering and validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
    /// Greater than
    GreaterThan,
    /// Greater than or equal to
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal to
    LessThanOrEqual,
    /// Contains (for strings/arrays)
    Contains,
    /// Does not contain
    DoesNotContain,
    /// Starts with (for strings)
    StartsWith,
    /// Ends with (for strings)
    EndsWith,
    /// Matches regex pattern
    Matches,
    /// In list of values
    In,
    /// Not in list of values
    NotIn,
    /// Is null/empty
    IsNull,
    /// Is not null/empty
    IsNotNull,
    /// Between two values (inclusive)
    Between,
    /// Custom comparison
    Custom(String),
}

/// Collection status enumeration
///
/// Tracks the current state of metric collection
/// for monitoring and management purposes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectionStatus {
    /// Collection is active and running
    Active,
    /// Collection is inactive/stopped
    Inactive,
    /// Collection is paused temporarily
    Paused,
    /// Collection encountered an error
    Error,
    /// Collection is in maintenance mode
    Maintenance,
    /// Collection is being configured
    Configuring,
    /// Collection is starting up
    Starting,
    /// Collection is shutting down
    Stopping,
}

/// Aggregation function types
///
/// Standard mathematical functions for metric aggregation
/// supporting both simple and complex analytical operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    /// Sum of all values
    Sum,
    /// Average of all values
    Average,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of data points
    Count,
    /// First value in time series
    First,
    /// Last value in time series
    Last,
    /// Median value
    Median,
    /// Standard deviation
    StdDev,
    /// Variance
    Variance,
    /// Specific percentile
    Percentile(f64),
    /// Rate of change
    Rate,
    /// Moving average
    MovingAverage { window: u64 },
    /// Custom aggregation function
    Custom(String),
}

/// Analysis types for metrics analytics
///
/// Defines different analytical operations that can be
/// performed on metric data for insights and monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisType {
    /// Trend analysis over time
    Trend,
    /// Seasonality detection
    Seasonality,
    /// Anomaly detection
    Anomaly,
    /// Correlation analysis
    Correlation,
    /// Forecasting
    Forecasting,
    /// Statistical summary
    Summary,
    /// Pattern recognition
    Pattern,
    /// Outlier detection
    OutlierDetection,
    /// Change point detection
    ChangePoint,
    /// Custom analysis type
    Custom(String),
}

/// Analysis result structure
///
/// Contains results from analytical operations
/// with confidence scores and detailed findings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Analysis type performed
    pub analysis_type: AnalysisType,
    /// Analysis timestamp
    pub timestamp: SystemTime,
    /// Analysis results
    pub results: HashMap<String, MetricValue>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Analysis summary
    pub summary: String,
    /// Detailed findings
    pub findings: Vec<AnalysisFinding>,
    /// Analysis metadata
    pub metadata: HashMap<String, String>,
}

/// Individual analysis finding
///
/// Represents a specific insight or pattern
/// discovered during metric analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisFinding {
    /// Finding type
    pub finding_type: String,
    /// Finding description
    pub description: String,
    /// Severity level
    pub severity: FindingSeverity,
    /// Confidence in finding (0.0 - 1.0)
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: HashMap<String, MetricValue>,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Finding severity levels
///
/// Categorizes the importance and urgency
/// of analytical findings for appropriate response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingSeverity {
    /// Informational finding
    Info,
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity requiring immediate attention
    Critical,
}

/// Utility functions for metric operations
impl MetricValue {
    /// Convert metric value to f64 if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetricValue::Integer(i) => Some(*i as f64),
            MetricValue::Float(f) => Some(*f),
            MetricValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Convert metric value to string representation
    pub fn as_string(&self) -> String {
        match self {
            MetricValue::Integer(i) => i.to_string(),
            MetricValue::Float(f) => f.to_string(),
            MetricValue::Boolean(b) => b.to_string(),
            MetricValue::String(s) => s.clone(),
            MetricValue::Duration(d) => format!("{}ms", d.as_millis()),
            MetricValue::Bytes(b) => format!("{} bytes", b.len()),
            MetricValue::Array(a) => format!("Array[{}]", a.len()),
            MetricValue::Object(o) => format!("Object{{{}}}", o.len()),
        }
    }

    /// Check if metric value is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self, MetricValue::Integer(_) | MetricValue::Float(_))
    }

    /// Get the size of the metric value in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        match self {
            MetricValue::Integer(_) => 8,
            MetricValue::Float(_) => 8,
            MetricValue::Boolean(_) => 1,
            MetricValue::String(s) => s.len(),
            MetricValue::Duration(_) => 16,
            MetricValue::Bytes(b) => b.len(),
            MetricValue::Array(a) => a.iter().map(|v| v.size_bytes()).sum(),
            MetricValue::Object(o) => {
                o.iter().map(|(k, v)| k.len() + v.size_bytes()).sum()
            }
        }
    }
}

impl MetricDataPoint {
    /// Create a new metric data point with current timestamp
    pub fn new(value: MetricValue, source: String) -> Self {
        Self {
            timestamp: SystemTime::now(),
            value,
            tags: HashMap::new(),
            metadata: MetricMetadata {
                source,
                quality: DataQuality::Unknown,
                confidence: 1.0,
                collection_method: CollectionMethod::Custom("default".to_string()),
                processing_info: ProcessingInfo {
                    pipeline_id: "default".to_string(),
                    stages: Vec::new(),
                    processed_at: SystemTime::now(),
                    processing_duration: Duration::from_millis(0),
                    processing_version: "1.0".to_string(),
                },
                validation_status: ValidationStatus::Pending,
            },
        }
    }

    /// Add a tag to the data point
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Set data quality
    pub fn with_quality(mut self, quality: DataQuality) -> Self {
        self.metadata.quality = quality;
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.metadata.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Check if data point is valid based on basic criteria
    pub fn is_valid(&self) -> bool {
        match &self.metadata.validation_status {
            ValidationStatus::Valid => true,
            ValidationStatus::Warning(_) => true,
            _ => false,
        }
    }

    /// Get age of data point
    pub fn age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.timestamp)
            .unwrap_or(Duration::from_secs(0))
    }
}

impl Default for DataQuality {
    fn default() -> Self {
        Self::Unknown
    }
}

impl Default for CollectionMethod {
    fn default() -> Self {
        Self::Pull
    }
}

impl Default for ValidationStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl Default for CollectionStatus {
    fn default() -> Self {
        Self::Inactive
    }
}

/// Standard metric tags used across the system
pub mod standard_tags {
    /// Host identifier tag
    pub const HOST: &str = "host";
    /// Service name tag
    pub const SERVICE: &str = "service";
    /// Environment tag (prod, staging, dev)
    pub const ENVIRONMENT: &str = "environment";
    /// Region tag
    pub const REGION: &str = "region";
    /// Version tag
    pub const VERSION: &str = "version";
    /// Component tag
    pub const COMPONENT: &str = "component";
    /// Instance identifier tag
    pub const INSTANCE: &str = "instance";
    /// Cluster identifier tag
    pub const CLUSTER: &str = "cluster";
    /// Namespace tag
    pub const NAMESPACE: &str = "namespace";
    /// Application tag
    pub const APPLICATION: &str = "application";
}

/// Common metric validation functions
pub mod validation {
    use super::*;

    /// Validate that numeric value is within bounds
    pub fn validate_numeric_bounds(
        value: &MetricValue,
        min: Option<f64>,
        max: Option<f64>,
    ) -> Result<(), String> {
        if let Some(num_value) = value.as_f64() {
            if let Some(min_val) = min {
                if num_value < min_val {
                    return Err(format!("Value {} is below minimum {}", num_value, min_val));
                }
            }
            if let Some(max_val) = max {
                if num_value > max_val {
                    return Err(format!("Value {} is above maximum {}", num_value, max_val));
                }
            }
            Ok(())
        } else {
            Err("Value is not numeric".to_string())
        }
    }

    /// Validate that string value matches pattern
    pub fn validate_string_pattern(value: &MetricValue, pattern: &str) -> Result<(), String> {
        if let MetricValue::String(s) = value {
            if regex::Regex::new(pattern)
                .map_err(|e| format!("Invalid pattern: {}", e))?
                .is_match(s)
            {
                Ok(())
            } else {
                Err(format!("String '{}' does not match pattern '{}'", s, pattern))
            }
        } else {
            Err("Value is not a string".to_string())
        }
    }

    /// Validate required tags are present
    pub fn validate_required_tags(
        tags: &HashMap<String, String>,
        required: &[String],
    ) -> Result<(), String> {
        for tag in required {
            if !tags.contains_key(tag) {
                return Err(format!("Required tag '{}' is missing", tag));
            }
        }
        Ok(())
    }
}

/// Metric units
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricUnit {
    None,
    Count,
    Bytes,
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
    Percent,
    Ratio,
    Rate(RateUnit),
    Custom(String),
}

/// Rate units
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateUnit {
    PerSecond,
    PerMinute,
    PerHour,
    PerDay,
    Custom(String),
}

/// Definition metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefinitionMetadata {
    pub created_at: SystemTime,
    pub created_by: String,
    pub modified_at: SystemTime,
    pub modified_by: String,
    pub version: String,
    pub tags: Vec<String>,
    pub category: String,
    pub documentation: Option<String>,
    pub examples: Vec<String>,
}

/// Metric collection state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCollectionState {
    pub state_id: String,
    pub metric_id: String,
    pub status: CollectionStatus,
    pub last_collection: Option<SystemTime>,
    pub next_collection: Option<SystemTime>,
    pub collection_count: u64,
    pub error_count: u64,
    pub last_error: Option<String>,
    pub performance_stats: CollectionPerformanceStats,
}


/// Collection performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionPerformanceStats {
    pub average_collection_time: Duration,
    pub success_rate: f64,
    pub throughput: f64,
    pub data_quality_score: f64,
    pub resource_usage: ResourceUsage,
}

/// Resource usage for collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub network_bytes: u64,
    pub disk_operations: u32,
    pub database_connections: u32,
}

/// Metric definition - forward declaration with simplified dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    pub metric_id: String,
    pub name: String,
    pub description: String,
    pub metric_type: MetricType,
    pub unit: MetricUnit,
    pub labels: Vec<String>,
    /// Note: These will be properly typed when modules are compiled together
    #[serde(skip)]
    pub collection_config: Option<()>,
    #[serde(skip)]
    pub aggregation_config: Option<()>,
    #[serde(skip)]
    pub retention_config: Option<()>,
    #[serde(skip)]
    pub alerting_config: Option<()>,
    #[serde(skip)]
    pub export_config: Option<()>,
    #[serde(skip)]
    pub validation_rules: Vec<()>,
    #[serde(skip)]
    pub transformation_rules: Vec<()>,
    pub metadata: DefinitionMetadata,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_value_conversions() {
        let int_value = MetricValue::Integer(42);
        assert_eq!(int_value.as_f64(), Some(42.0));
        assert!(int_value.is_numeric());
        assert_eq!(int_value.as_string(), "42");

        let string_value = MetricValue::String("test".to_string());
        assert_eq!(string_value.as_f64(), None);
        assert!(!string_value.is_numeric());
        assert_eq!(string_value.as_string(), "test");
    }

    #[test]
    fn test_metric_data_point_creation() {
        let data_point = MetricDataPoint::new(
            MetricValue::Float(3.14),
            "test_source".to_string(),
        )
        .with_tag("env".to_string(), "test".to_string())
        .with_quality(DataQuality::High)
        .with_confidence(0.95);

        assert_eq!(data_point.metadata.confidence, 0.95);
        assert!(matches!(data_point.metadata.quality, DataQuality::High));
        assert_eq!(data_point.tags.get("env"), Some(&"test".to_string()));
    }

    #[test]
    fn test_validation_functions() {
        let value = MetricValue::Float(5.0);

        // Test bounds validation
        assert!(validation::validate_numeric_bounds(&value, Some(0.0), Some(10.0)).is_ok());
        assert!(validation::validate_numeric_bounds(&value, Some(10.0), None).is_err());

        // Test required tags
        let mut tags = HashMap::new();
        tags.insert("host".to_string(), "server1".to_string());

        let required = vec!["host".to_string()];
        assert!(validation::validate_required_tags(&tags, &required).is_ok());

        let required = vec!["host".to_string(), "service".to_string()];
        assert!(validation::validate_required_tags(&tags, &required).is_err());
    }
}