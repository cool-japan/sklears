//! Performance Metrics and Data Management
//!
//! This module provides comprehensive performance metrics collection, storage, aggregation,
//! and analysis capabilities for the execution monitoring framework. It includes real-time
//! metrics, historical data, statistical analysis, and data quality management.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use sklears_core::error::{Result as SklResult, SklearsError};
use crate::monitoring_config::*;

/// Performance metric for monitoring
///
/// Represents a single performance measurement with timestamp, value,
/// and associated metadata for comprehensive performance tracking.
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    /// Unique metric identifier
    pub metric_id: String,

    /// Metric name
    pub name: String,

    /// Metric value
    pub value: f64,

    /// Unit of measurement
    pub unit: String,

    /// Measurement timestamp
    pub timestamp: SystemTime,

    /// Metric tags for categorization
    pub tags: HashMap<String, String>,

    /// Metric labels for grouping
    pub labels: HashMap<String, String>,

    /// Metric metadata
    pub metadata: MetricMetadata,

    /// Data quality indicators
    pub quality: DataQuality,
}

/// Metric metadata
#[derive(Debug, Clone)]
pub struct MetricMetadata {
    /// Source component that generated the metric
    pub source: String,

    /// Metric type classification
    pub metric_type: MetricClassification,

    /// Sampling information
    pub sampling_info: SamplingInfo,

    /// Aggregation level
    pub aggregation_level: AggregationLevel,

    /// Metric priority
    pub priority: MetricPriority,

    /// Retention policy for this metric
    pub retention_policy: Option<String>,
}

/// Metric classification
#[derive(Debug, Clone)]
pub enum MetricClassification {
    /// Counter metric (monotonically increasing)
    Counter,
    /// Gauge metric (arbitrary value)
    Gauge,
    /// Histogram metric (distribution of values)
    Histogram,
    /// Summary metric (quantiles over sliding time window)
    Summary,
    /// Timer metric (timing measurements)
    Timer,
    /// Rate metric (rate of events)
    Rate,
    /// Ratio metric (ratio between two values)
    Ratio,
}

/// Sampling information
#[derive(Debug, Clone)]
pub struct SamplingInfo {
    /// Sampling rate used
    pub rate: f64,

    /// Sampling method
    pub method: SamplingMethod,

    /// Original sample count
    pub original_count: u64,

    /// Sampled count
    pub sampled_count: u64,
}

/// Sampling methods
#[derive(Debug, Clone)]
pub enum SamplingMethod {
    Random,
    Systematic,
    Stratified,
    Reservoir,
    Adaptive,
}

/// Aggregation levels
#[derive(Debug, Clone)]
pub enum AggregationLevel {
    Raw,
    Second,
    Minute,
    Hour,
    Day,
    Custom(Duration),
}

/// Metric priority levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum MetricPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Data quality indicators
#[derive(Debug, Clone)]
pub struct DataQuality {
    /// Completeness (0.0 to 1.0)
    pub completeness: f64,

    /// Accuracy confidence (0.0 to 1.0)
    pub accuracy: f64,

    /// Data freshness
    pub freshness: Duration,

    /// Consistency score (0.0 to 1.0)
    pub consistency: f64,

    /// Validation status
    pub validation_status: ValidationStatus,
}

/// Validation status
#[derive(Debug, Clone)]
pub enum ValidationStatus {
    Valid,
    Warning { issues: Vec<String> },
    Invalid { errors: Vec<String> },
    Unknown,
}

/// Metric data point for time series
#[derive(Debug, Clone)]
pub struct MetricDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,

    /// Metric value
    pub value: f64,

    /// Data quality for this point
    pub quality: DataQuality,

    /// Additional annotations
    pub annotations: HashMap<String, String>,
}

/// Metric statistics
#[derive(Debug, Clone)]
pub struct MetricStatistics {
    /// Count of data points
    pub count: usize,

    /// Mean value
    pub mean: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Median value
    pub median: f64,

    /// Percentiles
    pub percentiles: HashMap<String, f64>,

    /// Variance
    pub variance: f64,

    /// Skewness
    pub skewness: f64,

    /// Kurtosis
    pub kurtosis: f64,
}

/// Historical metric data
#[derive(Debug, Clone)]
pub struct HistoricalMetric {
    /// Metric name
    pub name: String,

    /// Time series data points
    pub data_points: Vec<MetricDataPoint>,

    /// Aggregated statistics
    pub statistics: MetricStatistics,

    /// Trend information
    pub trend: TrendInfo,

    /// Seasonality information
    pub seasonality: SeasonalityInfo,
}

/// Trend information
#[derive(Debug, Clone)]
pub struct TrendInfo {
    /// Trend direction
    pub direction: TrendDirection,

    /// Trend strength (0.0 to 1.0)
    pub strength: f64,

    /// Trend slope
    pub slope: f64,

    /// Trend confidence (0.0 to 1.0)
    pub confidence: f64,

    /// Change points
    pub change_points: Vec<SystemTime>,
}

/// Trend directions
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Unknown,
}

/// Seasonality information
#[derive(Debug, Clone)]
pub struct SeasonalityInfo {
    /// Seasonal patterns detected
    pub patterns: Vec<SeasonalPattern>,

    /// Seasonal strength (0.0 to 1.0)
    pub strength: f64,

    /// Seasonal period
    pub period: Option<Duration>,
}

/// Seasonal pattern
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    /// Pattern period
    pub period: Duration,

    /// Pattern amplitude
    pub amplitude: f64,

    /// Pattern phase
    pub phase: f64,

    /// Pattern confidence
    pub confidence: f64,
}

/// Metrics storage trait
pub trait MetricsStorage: Send + Sync {
    /// Store a metric
    fn store_metric(&mut self, session_id: &str, metric: &PerformanceMetric) -> SklResult<()>;

    /// Retrieve metrics within time range
    fn retrieve_metrics(&self, session_id: &str, time_range: &TimeRange) -> SklResult<Vec<PerformanceMetric>>;

    /// Retrieve metrics by name
    fn retrieve_metrics_by_name(&self, session_id: &str, metric_name: &str, time_range: &TimeRange) -> SklResult<Vec<PerformanceMetric>>;

    /// Aggregate metrics according to configuration
    fn aggregate_metrics(&self, session_id: &str, aggregation: &MetricsAggregationConfig) -> SklResult<Vec<PerformanceMetric>>;

    /// Get metric statistics
    fn get_statistics(&self, session_id: &str, metric_name: &str, time_range: &TimeRange) -> SklResult<MetricStatistics>;

    /// Delete metrics older than specified time
    fn cleanup_old_metrics(&mut self, cutoff_time: SystemTime) -> SklResult<usize>;

    /// Get storage health information
    fn get_health(&self) -> SklResult<StorageHealth>;
}

/// Storage health information
#[derive(Debug, Clone)]
pub struct StorageHealth {
    /// Storage status
    pub status: StorageStatus,

    /// Used capacity
    pub used_capacity: u64,

    /// Total capacity
    pub total_capacity: u64,

    /// Number of stored metrics
    pub metric_count: u64,

    /// Storage performance metrics
    pub performance: StoragePerformance,
}

/// Storage status
#[derive(Debug, Clone)]
pub enum StorageStatus {
    Healthy,
    Warning,
    Critical,
    Unavailable,
}

/// Storage performance metrics
#[derive(Debug, Clone)]
pub struct StoragePerformance {
    /// Average write latency
    pub avg_write_latency: Duration,

    /// Average read latency
    pub avg_read_latency: Duration,

    /// Write throughput (metrics per second)
    pub write_throughput: f64,

    /// Read throughput (metrics per second)
    pub read_throughput: f64,

    /// Error rate
    pub error_rate: f64,
}

/// In-memory metrics storage implementation
#[derive(Debug)]
pub struct InMemoryMetricsStorage {
    /// Storage for metrics by session
    storage: Arc<RwLock<HashMap<String, VecDeque<PerformanceMetric>>>>,

    /// Storage configuration
    config: MetricsStorageConfig,

    /// Storage statistics
    stats: Arc<RwLock<StorageStats>>,
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total metrics stored
    pub total_metrics: u64,

    /// Total storage operations
    pub total_operations: u64,

    /// Failed operations
    pub failed_operations: u64,

    /// Average operation time
    pub avg_operation_time: Duration,

    /// Current memory usage
    pub memory_usage: u64,
}

impl InMemoryMetricsStorage {
    /// Create new in-memory storage
    pub fn new(config: MetricsStorageConfig) -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(StorageStats::default())),
        }
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        self.stats.read().unwrap().clone()
    }
}

impl MetricsStorage for InMemoryMetricsStorage {
    fn store_metric(&mut self, session_id: &str, metric: &PerformanceMetric) -> SklResult<()> {
        let start_time = Instant::now();
        let mut storage = self.storage.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let session_metrics = storage.entry(session_id.to_string()).or_insert_with(VecDeque::new);

        // Check capacity limits
        if session_metrics.len() >= self.config.capacity.max_entries {
            match self.config.capacity.cleanup_policy {
                CleanupPolicy::RemoveOldest => {
                    session_metrics.pop_front();
                }
                CleanupPolicy::Priority => {
                    // Remove lowest priority metric (simplified)
                    if let Some(pos) = session_metrics.iter().position(|m| m.metadata.priority == MetricPriority::Low) {
                        session_metrics.remove(pos);
                    } else {
                        session_metrics.pop_front();
                    }
                }
                _ => {
                    // For other policies, just remove oldest
                    session_metrics.pop_front();
                }
            }
        }

        session_metrics.push_back(metric.clone());

        stats.total_metrics += 1;
        stats.total_operations += 1;
        stats.avg_operation_time = (stats.avg_operation_time * (stats.total_operations - 1) as u32 + start_time.elapsed()) / stats.total_operations as u32;

        Ok(())
    }

    fn retrieve_metrics(&self, session_id: &str, time_range: &TimeRange) -> SklResult<Vec<PerformanceMetric>> {
        let storage = self.storage.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        stats.total_operations += 1;

        if let Some(metrics) = storage.get(session_id) {
            let filtered: Vec<PerformanceMetric> = metrics
                .iter()
                .filter(|metric| time_range.contains(metric.timestamp))
                .cloned()
                .collect();
            Ok(filtered)
        } else {
            Ok(Vec::new())
        }
    }

    fn retrieve_metrics_by_name(&self, session_id: &str, metric_name: &str, time_range: &TimeRange) -> SklResult<Vec<PerformanceMetric>> {
        let storage = self.storage.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        stats.total_operations += 1;

        if let Some(metrics) = storage.get(session_id) {
            let filtered: Vec<PerformanceMetric> = metrics
                .iter()
                .filter(|metric| metric.name == metric_name && time_range.contains(metric.timestamp))
                .cloned()
                .collect();
            Ok(filtered)
        } else {
            Ok(Vec::new())
        }
    }

    fn aggregate_metrics(&self, session_id: &str, aggregation: &MetricsAggregationConfig) -> SklResult<Vec<PerformanceMetric>> {
        let metrics = self.retrieve_metrics(session_id, &TimeRange {
            start: SystemTime::now() - aggregation.window_size,
            end: SystemTime::now(),
        })?;

        let mut aggregated = Vec::new();

        // Group metrics by name
        let mut grouped: HashMap<String, Vec<&PerformanceMetric>> = HashMap::new();
        for metric in &metrics {
            grouped.entry(metric.name.clone()).or_insert_with(Vec::new).push(metric);
        }

        // Apply aggregation functions
        for (name, metric_group) in grouped {
            for function in &aggregation.functions {
                let aggregated_value = match function {
                    AggregationFunction::Mean => {
                        metric_group.iter().map(|m| m.value).sum::<f64>() / metric_group.len() as f64
                    }
                    AggregationFunction::Sum => {
                        metric_group.iter().map(|m| m.value).sum::<f64>()
                    }
                    AggregationFunction::Min => {
                        metric_group.iter().map(|m| m.value).fold(f64::INFINITY, f64::min)
                    }
                    AggregationFunction::Max => {
                        metric_group.iter().map(|m| m.value).fold(f64::NEG_INFINITY, f64::max)
                    }
                    AggregationFunction::Count => {
                        metric_group.len() as f64
                    }
                    _ => continue, // Skip unsupported functions for simplicity
                };

                let aggregated_metric = PerformanceMetric {
                    metric_id: format!("{}_{:?}", name, function),
                    name: format!("{}_{:?}", name, function),
                    value: aggregated_value,
                    unit: metric_group[0].unit.clone(),
                    timestamp: SystemTime::now(),
                    tags: metric_group[0].tags.clone(),
                    labels: metric_group[0].labels.clone(),
                    metadata: MetricMetadata {
                        source: "aggregator".to_string(),
                        metric_type: MetricClassification::Gauge,
                        sampling_info: SamplingInfo {
                            rate: 1.0,
                            method: SamplingMethod::Systematic,
                            original_count: metric_group.len() as u64,
                            sampled_count: metric_group.len() as u64,
                        },
                        aggregation_level: AggregationLevel::Custom(aggregation.window_size),
                        priority: MetricPriority::Normal,
                        retention_policy: None,
                    },
                    quality: DataQuality::default(),
                };

                aggregated.push(aggregated_metric);
            }
        }

        Ok(aggregated)
    }

    fn get_statistics(&self, session_id: &str, metric_name: &str, time_range: &TimeRange) -> SklResult<MetricStatistics> {
        let metrics = self.retrieve_metrics_by_name(session_id, metric_name, time_range)?;

        if metrics.is_empty() {
            return Err(SklearsError::InvalidInput("No metrics found for specified criteria".to_string()));
        }

        let values: Vec<f64> = metrics.iter().map(|m| m.value).collect();
        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_values[0];
        let max = sorted_values[count - 1];
        let median = if count % 2 == 0 {
            (sorted_values[count / 2 - 1] + sorted_values[count / 2]) / 2.0
        } else {
            sorted_values[count / 2]
        };

        // Calculate percentiles
        let mut percentiles = HashMap::new();
        for &p in &[0.25, 0.50, 0.75, 0.90, 0.95, 0.99] {
            let index = ((count as f64 - 1.0) * p) as usize;
            percentiles.insert(format!("p{}", (p * 100.0) as u32), sorted_values[index]);
        }

        // Calculate skewness and kurtosis (simplified)
        let skewness = values.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / count as f64;
        let kurtosis = values.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / count as f64 - 3.0;

        Ok(MetricStatistics {
            count,
            mean,
            std_dev,
            min,
            max,
            median,
            percentiles,
            variance,
            skewness,
            kurtosis,
        })
    }

    fn cleanup_old_metrics(&mut self, cutoff_time: SystemTime) -> SklResult<usize> {
        let mut storage = self.storage.write().unwrap();
        let mut total_removed = 0;

        for (_session_id, metrics) in storage.iter_mut() {
            let initial_len = metrics.len();
            metrics.retain(|metric| metric.timestamp >= cutoff_time);
            total_removed += initial_len - metrics.len();
        }

        Ok(total_removed)
    }

    fn get_health(&self) -> SklResult<StorageHealth> {
        let stats = self.get_stats();
        let memory_usage = stats.memory_usage;

        let status = if memory_usage < self.config.capacity.max_size / 2 {
            StorageStatus::Healthy
        } else if memory_usage < (self.config.capacity.max_size * 8) / 10 {
            StorageStatus::Warning
        } else {
            StorageStatus::Critical
        };

        Ok(StorageHealth {
            status,
            used_capacity: memory_usage,
            total_capacity: self.config.capacity.max_size,
            metric_count: stats.total_metrics,
            performance: StoragePerformance {
                avg_write_latency: stats.avg_operation_time,
                avg_read_latency: stats.avg_operation_time,
                write_throughput: 1000.0, // Placeholder
                read_throughput: 2000.0,  // Placeholder
                error_rate: stats.failed_operations as f64 / stats.total_operations as f64,
            },
        })
    }
}

impl Default for StorageStats {
    fn default() -> Self {
        Self {
            total_metrics: 0,
            total_operations: 0,
            failed_operations: 0,
            avg_operation_time: Duration::ZERO,
            memory_usage: 0,
        }
    }
}

/// Metrics collector for gathering performance metrics
#[derive(Debug)]
pub struct MetricsCollector {
    /// Storage backend
    storage: Box<dyn MetricsStorage>,

    /// Collection configuration
    config: MetricsConfig,

    /// Active sessions
    sessions: HashMap<String, SessionMetrics>,

    /// Collector statistics
    stats: Arc<RwLock<CollectorStats>>,
}

/// Session-specific metrics
#[derive(Debug)]
pub struct SessionMetrics {
    /// Session ID
    pub session_id: String,

    /// Session start time
    pub start_time: SystemTime,

    /// Metrics collected count
    pub metrics_count: u64,

    /// Last metric timestamp
    pub last_metric_time: Option<SystemTime>,

    /// Session-specific configuration
    pub config: MetricsConfig,
}

/// Collector statistics
#[derive(Debug, Clone)]
pub struct CollectorStats {
    /// Total metrics collected
    pub total_collected: u64,

    /// Collection rate (metrics per second)
    pub collection_rate: f64,

    /// Failed collections
    pub failed_collections: u64,

    /// Average collection time
    pub avg_collection_time: Duration,

    /// Active sessions count
    pub active_sessions: usize,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(config: MetricsConfig) -> Self {
        let storage = Box::new(InMemoryMetricsStorage::new(config.storage.clone()));

        Self {
            storage,
            config,
            sessions: HashMap::new(),
            stats: Arc::new(RwLock::new(CollectorStats::default())),
        }
    }

    /// Initialize session
    pub fn initialize_session(&mut self, session_id: &str, config: &MetricsConfig) -> SklResult<()> {
        let session_metrics = SessionMetrics {
            session_id: session_id.to_string(),
            start_time: SystemTime::now(),
            metrics_count: 0,
            last_metric_time: None,
            config: config.clone(),
        };

        self.sessions.insert(session_id.to_string(), session_metrics);
        self.stats.write().unwrap().active_sessions = self.sessions.len();

        Ok(())
    }

    /// Collect metric
    pub fn collect_metric(&mut self, session_id: &str, metric: PerformanceMetric) -> SklResult<()> {
        let start_time = Instant::now();

        // Store metric
        self.storage.store_metric(session_id, &metric)?;

        // Update session metrics
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.metrics_count += 1;
            session.last_metric_time = Some(metric.timestamp);
        }

        // Update collector stats
        let mut stats = self.stats.write().unwrap();
        stats.total_collected += 1;
        stats.avg_collection_time = (stats.avg_collection_time * (stats.total_collected - 1) as u32 + start_time.elapsed()) / stats.total_collected as u32;

        Ok(())
    }

    /// Get real-time metrics for a session
    pub fn get_real_time_metrics(&self, session_id: &str) -> SklResult<Vec<PerformanceMetric>> {
        let time_range = TimeRange {
            start: SystemTime::now() - Duration::from_secs(60), // Last minute
            end: SystemTime::now(),
        };

        self.storage.retrieve_metrics(session_id, &time_range)
    }

    /// Finalize session
    pub fn finalize_session(&mut self, session_id: &str) -> SklResult<Vec<PerformanceMetric>> {
        // Get all metrics for the session
        let time_range = TimeRange {
            start: SystemTime::UNIX_EPOCH,
            end: SystemTime::now(),
        };

        let metrics = self.storage.retrieve_metrics(session_id, &time_range)?;

        // Remove session
        self.sessions.remove(session_id);
        self.stats.write().unwrap().active_sessions = self.sessions.len();

        Ok(metrics)
    }

    /// Get collector statistics
    pub fn get_stats(&self) -> CollectorStats {
        self.stats.read().unwrap().clone()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MetricsConfig) -> SklResult<()> {
        self.config = config;
        Ok(())
    }

    /// Get health status
    pub fn get_health_status(&self) -> SklResult<crate::monitoring_core::ComponentHealth> {
        let storage_health = self.storage.get_health()?;
        let stats = self.get_stats();

        let mut health_score = 1.0;
        let mut issues = Vec::new();

        // Check storage health
        match storage_health.status {
            StorageStatus::Warning => {
                health_score -= 0.2;
                issues.push("Storage capacity warning".to_string());
            }
            StorageStatus::Critical => {
                health_score -= 0.5;
                issues.push("Storage capacity critical".to_string());
            }
            StorageStatus::Unavailable => {
                health_score = 0.0;
                issues.push("Storage unavailable".to_string());
            }
            _ => {}
        }

        // Check collection rate
        if stats.failed_collections > 0 {
            health_score -= 0.3;
            issues.push("Collection failures detected".to_string());
        }

        let status = if health_score >= 0.8 {
            crate::monitoring_core::HealthStatus::Healthy
        } else if health_score >= 0.5 {
            crate::monitoring_core::HealthStatus::Warning
        } else {
            crate::monitoring_core::HealthStatus::Critical
        };

        Ok(crate::monitoring_core::ComponentHealth {
            component: "metrics_collector".to_string(),
            status,
            score: health_score,
            last_check: SystemTime::now(),
            issues,
        })
    }
}

impl Default for CollectorStats {
    fn default() -> Self {
        Self {
            total_collected: 0,
            collection_rate: 0.0,
            failed_collections: 0,
            avg_collection_time: Duration::ZERO,
            active_sessions: 0,
        }
    }
}

/// Metrics collection system
pub struct MetricsCollectionSystem {
    /// Metrics collector
    collector: MetricsCollector,

    /// System configuration
    config: MetricsConfig,
}

impl MetricsCollectionSystem {
    /// Create new metrics collection system
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            collector: MetricsCollector::new(config.clone()),
            config,
        }
    }

    /// Initialize session
    pub fn initialize_session(&mut self, session_id: &str, config: &MetricsConfig) -> SklResult<()> {
        self.collector.initialize_session(session_id, config)
    }

    /// Get real-time metrics
    pub fn get_real_time_metrics(&self, session_id: &str) -> SklResult<Vec<PerformanceMetric>> {
        self.collector.get_real_time_metrics(session_id)
    }

    /// Finalize session
    pub fn finalize_session(&mut self, session_id: &str) -> SklResult<Vec<PerformanceMetric>> {
        self.collector.finalize_session(session_id)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MetricsConfig) -> SklResult<()> {
        self.config = config.clone();
        self.collector.update_config(config)
    }

    /// Get health status
    pub fn get_health_status(&self) -> SklResult<crate::monitoring_core::ComponentHealth> {
        self.collector.get_health_status()
    }
}

impl Default for DataQuality {
    fn default() -> Self {
        Self {
            completeness: 1.0,
            accuracy: 1.0,
            freshness: Duration::ZERO,
            consistency: 1.0,
            validation_status: ValidationStatus::Valid,
        }
    }
}

/// Time range specification
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Start time
    pub start: SystemTime,
    /// End time
    pub end: SystemTime,
}

impl TimeRange {
    /// Check if timestamp is within range
    pub fn contains(&self, timestamp: SystemTime) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }
}

// Utility functions for metric creation

impl PerformanceMetric {
    /// Create new performance metric
    pub fn new(name: String, value: f64, unit: String) -> Self {
        Self {
            metric_id: uuid::Uuid::new_v4().to_string(),
            name,
            value,
            unit,
            timestamp: SystemTime::now(),
            tags: HashMap::new(),
            labels: HashMap::new(),
            metadata: MetricMetadata::default(),
            quality: DataQuality::default(),
        }
    }

    /// Add tag to metric
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Add label to metric
    pub fn with_label(mut self, key: String, value: String) -> Self {
        self.labels.insert(key, value);
        self
    }

    /// Set metric priority
    pub fn with_priority(mut self, priority: MetricPriority) -> Self {
        self.metadata.priority = priority;
        self
    }
}

impl Default for MetricMetadata {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            metric_type: MetricClassification::Gauge,
            sampling_info: SamplingInfo {
                rate: 1.0,
                method: SamplingMethod::Systematic,
                original_count: 1,
                sampled_count: 1,
            },
            aggregation_level: AggregationLevel::Raw,
            priority: MetricPriority::Normal,
            retention_policy: None,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_creation() {
        let metric = PerformanceMetric::new(
            "cpu_usage".to_string(),
            75.5,
            "percent".to_string(),
        )
        .with_tag("host".to_string(), "server1".to_string())
        .with_priority(MetricPriority::High);

        assert_eq!(metric.name, "cpu_usage");
        assert_eq!(metric.value, 75.5);
        assert_eq!(metric.unit, "percent");
        assert_eq!(metric.metadata.priority, MetricPriority::High);
        assert_eq!(metric.tags.get("host").unwrap(), "server1");
    }

    #[test]
    fn test_in_memory_storage() {
        let config = MetricsStorageConfig::default();
        let mut storage = InMemoryMetricsStorage::new(config);

        let metric = PerformanceMetric::new(
            "test_metric".to_string(),
            42.0,
            "units".to_string(),
        );

        storage.store_metric("session1", &metric).unwrap();

        let time_range = TimeRange {
            start: SystemTime::now() - Duration::from_secs(60),
            end: SystemTime::now() + Duration::from_secs(60),
        };

        let retrieved = storage.retrieve_metrics("session1", &time_range).unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].name, "test_metric");
    }

    #[test]
    fn test_metrics_statistics() {
        let config = MetricsStorageConfig::default();
        let mut storage = InMemoryMetricsStorage::new(config);

        // Store multiple metrics
        for i in 0..10 {
            let metric = PerformanceMetric::new(
                "test_metric".to_string(),
                i as f64,
                "units".to_string(),
            );
            storage.store_metric("session1", &metric).unwrap();
        }

        let time_range = TimeRange {
            start: SystemTime::now() - Duration::from_secs(60),
            end: SystemTime::now() + Duration::from_secs(60),
        };

        let stats = storage.get_statistics("session1", "test_metric", &time_range).unwrap();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.mean, 4.5);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 9.0);
    }

    #[test]
    fn test_metrics_collector() {
        let config = MetricsConfig::default();
        let mut collector = MetricsCollector::new(config.clone());

        collector.initialize_session("session1", &config).unwrap();

        let metric = PerformanceMetric::new(
            "test_metric".to_string(),
            100.0,
            "percent".to_string(),
        );

        collector.collect_metric("session1", metric).unwrap();

        let stats = collector.get_stats();
        assert_eq!(stats.total_collected, 1);
        assert_eq!(stats.active_sessions, 1);
    }

    #[test]
    fn test_data_quality() {
        let quality = DataQuality::default();
        assert_eq!(quality.completeness, 1.0);
        assert_eq!(quality.accuracy, 1.0);
        assert!(matches!(quality.validation_status, ValidationStatus::Valid));
    }

    #[test]
    fn test_time_range() {
        let start = SystemTime::now();
        let end = start + Duration::from_secs(60);
        let range = TimeRange { start, end };

        let within = start + Duration::from_secs(30);
        let outside = start + Duration::from_secs(90);

        assert!(range.contains(within));
        assert!(!range.contains(outside));
    }
}