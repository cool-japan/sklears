//! Metrics Collection and Storage
//!
//! This module provides comprehensive performance metrics collection, storage,
//! and aggregation capabilities for monitoring execution performance, resource
//! utilization, and system behavior patterns.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;

use crate::monitoring_core::TimeRange;
use crate::configuration_management::MetricsCollectionConfig;

/// Performance metric with value, timestamp, and metadata
///
/// Represents a single performance measurement with comprehensive metadata
/// for analysis, alerting, and trend tracking.
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    /// Metric name/identifier
    pub name: String,

    /// Metric value
    pub value: f64,

    /// Unit of measurement
    pub unit: String,

    /// Timestamp when metric was collected
    pub timestamp: SystemTime,

    /// Metric tags for categorization and filtering
    pub tags: HashMap<String, String>,

    /// Metric type classification
    pub metric_type: MetricType,

    /// Collection source information
    pub source: MetricSource,

    /// Metric quality indicators
    pub quality: MetricQuality,

    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl PerformanceMetric {
    /// Create a new performance metric
    pub fn new(name: String, value: f64, unit: String) -> Self {
        Self {
            name,
            value,
            unit,
            timestamp: SystemTime::now(),
            tags: HashMap::new(),
            metric_type: MetricType::Gauge,
            source: MetricSource::System,
            quality: MetricQuality::default(),
            metadata: HashMap::new(),
        }
    }

    /// Builder pattern for metric creation
    pub fn builder() -> MetricBuilder {
        MetricBuilder::new()
    }

    /// Add tag to metric
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Set metric type
    pub fn with_type(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self
    }

    /// Set metric source
    pub fn with_source(mut self, source: MetricSource) -> Self {
        self.source = source;
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: SystemTime) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get age of metric from current time
    pub fn age(&self) -> Duration {
        SystemTime::now().duration_since(self.timestamp).unwrap_or(Duration::from_secs(0))
    }

    /// Check if metric matches tag filter
    pub fn matches_tags(&self, filter: &HashMap<String, String>) -> bool {
        filter.iter().all(|(key, value)| {
            self.tags.get(key).map_or(false, |v| v == value)
        })
    }

    /// Get metric identifier including tags
    pub fn identifier(&self) -> String {
        if self.tags.is_empty() {
            self.name.clone()
        } else {
            let mut tag_parts: Vec<String> = self.tags.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            tag_parts.sort();
            format!("{}[{}]", self.name, tag_parts.join(","))
        }
    }

    /// Check if metric is valid
    pub fn is_valid(&self) -> bool {
        !self.name.is_empty() &&
        self.value.is_finite() &&
        !self.unit.is_empty() &&
        self.quality.accuracy >= 0.0 &&
        self.quality.accuracy <= 1.0
    }

    /// Calculate normalized value for comparison
    pub fn normalized_value(&self, min_val: f64, max_val: f64) -> f64 {
        if max_val == min_val {
            0.0
        } else {
            (self.value - min_val) / (max_val - min_val)
        }
    }
}

/// Metric builder for fluent construction
#[derive(Debug)]
pub struct MetricBuilder {
    name: Option<String>,
    value: Option<f64>,
    unit: Option<String>,
    timestamp: Option<SystemTime>,
    tags: HashMap<String, String>,
    metric_type: MetricType,
    source: MetricSource,
    quality: MetricQuality,
    metadata: HashMap<String, String>,
}

impl MetricBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            name: None,
            value: None,
            unit: None,
            timestamp: None,
            tags: HashMap::new(),
            metric_type: MetricType::Gauge,
            source: MetricSource::System,
            quality: MetricQuality::default(),
            metadata: HashMap::new(),
        }
    }

    /// Set metric name
    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Set metric value
    pub fn value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Set metric unit
    pub fn unit(mut self, unit: String) -> Self {
        self.unit = Some(unit);
        self
    }

    /// Set timestamp
    pub fn timestamp(mut self, timestamp: SystemTime) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Add tag
    pub fn tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Set metric type
    pub fn metric_type(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self
    }

    /// Set source
    pub fn source(mut self, source: MetricSource) -> Self {
        self.source = source;
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Build the metric
    pub fn build(self) -> SklResult<PerformanceMetric> {
        let name = self.name.ok_or_else(|| SklearsError::InvalidInput("Metric name is required".to_string()))?;
        let value = self.value.ok_or_else(|| SklearsError::InvalidInput("Metric value is required".to_string()))?;
        let unit = self.unit.ok_or_else(|| SklearsError::InvalidInput("Metric unit is required".to_string()))?;

        Ok(PerformanceMetric {
            name,
            value,
            unit,
            timestamp: self.timestamp.unwrap_or_else(SystemTime::now),
            tags: self.tags,
            metric_type: self.metric_type,
            source: self.source,
            quality: self.quality,
            metadata: self.metadata,
        })
    }
}

impl Default for MetricBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of performance metrics
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    /// Gauge - instantaneous value that can go up or down
    Gauge,

    /// Counter - cumulative value that only increases
    Counter,

    /// Histogram - distribution of values over time
    Histogram,

    /// Timer - duration measurements
    Timer,

    /// Rate - rate of events per time unit
    Rate,

    /// Execution time measurements
    ExecutionTime,

    /// Resource utilization metrics
    ResourceUtilization,

    /// Throughput measurements
    Throughput,

    /// Error rate metrics
    ErrorRates,

    /// Queue depth metrics
    QueueDepth,

    /// Latency measurements
    Latency,

    /// Memory usage metrics
    MemoryUsage,

    /// CPU usage metrics
    CpuUsage,

    /// Network I/O metrics
    NetworkIO,

    /// Disk I/O metrics
    DiskIO,

    /// Custom metric type
    Custom { name: String },
}

impl MetricType {
    /// Check if metric type is a cumulative type
    pub fn is_cumulative(&self) -> bool {
        matches!(self, MetricType::Counter)
    }

    /// Check if metric type represents a rate
    pub fn is_rate(&self) -> bool {
        matches!(self, MetricType::Rate | MetricType::Throughput | MetricType::ErrorRates)
    }

    /// Check if metric type represents a duration
    pub fn is_duration(&self) -> bool {
        matches!(self, MetricType::Timer | MetricType::ExecutionTime | MetricType::Latency)
    }

    /// Get default unit for metric type
    pub fn default_unit(&self) -> &'static str {
        match self {
            MetricType::ExecutionTime | MetricType::Timer | MetricType::Latency => "milliseconds",
            MetricType::ResourceUtilization | MetricType::CpuUsage | MetricType::ErrorRates => "percentage",
            MetricType::Throughput => "operations_per_second",
            MetricType::MemoryUsage => "bytes",
            MetricType::NetworkIO | MetricType::DiskIO => "bytes_per_second",
            MetricType::QueueDepth => "count",
            MetricType::Counter => "count",
            MetricType::Rate => "per_second",
            _ => "value",
        }
    }
}

/// Metric source information
#[derive(Debug, Clone, PartialEq)]
pub enum MetricSource {
    /// System-level metrics
    System,

    /// Application-level metrics
    Application,

    /// User-defined metrics
    User,

    /// External system metrics
    External { system_name: String },

    /// Derived/calculated metrics
    Derived,

    /// Synthetic/test metrics
    Synthetic,
}

impl MetricSource {
    /// Get source reliability score (0.0 to 1.0)
    pub fn reliability_score(&self) -> f64 {
        match self {
            MetricSource::System => 0.95,
            MetricSource::Application => 0.9,
            MetricSource::User => 0.8,
            MetricSource::External { .. } => 0.85,
            MetricSource::Derived => 0.7,
            MetricSource::Synthetic => 0.6,
        }
    }
}

/// Metric quality indicators
#[derive(Debug, Clone)]
pub struct MetricQuality {
    /// Accuracy score (0.0 to 1.0)
    pub accuracy: f64,

    /// Completeness score (0.0 to 1.0)
    pub completeness: f64,

    /// Timeliness score (0.0 to 1.0)
    pub timeliness: f64,

    /// Consistency score (0.0 to 1.0)
    pub consistency: f64,

    /// Number of validation checks passed
    pub validation_checks_passed: u32,

    /// Number of validation checks failed
    pub validation_checks_failed: u32,
}

impl Default for MetricQuality {
    fn default() -> Self {
        Self {
            accuracy: 1.0,
            completeness: 1.0,
            timeliness: 1.0,
            consistency: 1.0,
            validation_checks_passed: 0,
            validation_checks_failed: 0,
        }
    }
}

impl MetricQuality {
    /// Calculate overall quality score
    pub fn overall_score(&self) -> f64 {
        let scores = [self.accuracy, self.completeness, self.timeliness, self.consistency];
        scores.iter().sum::<f64>() / scores.len() as f64
    }

    /// Check if quality meets threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.overall_score() >= threshold
    }
}

/// Metrics storage trait for different storage backends
pub trait MetricsStorage: Send + Sync {
    /// Store a metric
    fn store_metric(&mut self, session_id: &str, metric: &PerformanceMetric) -> SklResult<()>;

    /// Retrieve metrics for a session within time range
    fn retrieve_metrics(&self, session_id: &str, time_range: &TimeRange) -> SklResult<Vec<PerformanceMetric>>;

    /// Aggregate metrics according to configuration
    fn aggregate_metrics(&self, session_id: &str, aggregation: &MetricsAggregationConfig) -> SklResult<Vec<PerformanceMetric>>;

    /// Get metrics count for session
    fn metrics_count(&self, session_id: &str) -> SklResult<u64>;

    /// Delete metrics for session
    fn delete_metrics(&mut self, session_id: &str) -> SklResult<u64>;

    /// Get storage statistics
    fn storage_stats(&self) -> StorageStatistics;

    /// Cleanup old metrics
    fn cleanup_old_metrics(&mut self, older_than: Duration) -> SklResult<u64>;
}

/// In-memory metrics storage implementation
#[derive(Debug)]
pub struct InMemoryMetricsStorage {
    /// Storage by session ID
    storage: HashMap<String, Vec<PerformanceMetric>>,

    /// Storage lock for thread safety
    lock: Arc<RwLock<()>>,

    /// Storage configuration
    config: StorageConfig,

    /// Storage statistics
    stats: StorageStatistics,
}

impl InMemoryMetricsStorage {
    /// Create new in-memory storage
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
            lock: Arc::new(RwLock::new(())),
            config: StorageConfig::default(),
            stats: StorageStatistics::new(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: StorageConfig) -> Self {
        Self {
            storage: HashMap::new(),
            lock: Arc::new(RwLock::new(())),
            config,
            stats: StorageStatistics::new(),
        }
    }

    /// Check storage limits
    fn check_limits(&self, session_id: &str) -> SklResult<()> {
        if let Some(metrics) = self.storage.get(session_id) {
            if metrics.len() >= self.config.max_metrics_per_session {
                return Err(SklearsError::ResourceExhausted(
                    format!("Maximum metrics per session ({}) exceeded", self.config.max_metrics_per_session)
                ));
            }
        }

        let total_metrics: usize = self.storage.values().map(|v| v.len()).sum();
        if total_metrics >= self.config.max_total_metrics {
            return Err(SklearsError::ResourceExhausted(
                format!("Maximum total metrics ({}) exceeded", self.config.max_total_metrics)
            ));
        }

        Ok(())
    }
}

impl MetricsStorage for InMemoryMetricsStorage {
    fn store_metric(&mut self, session_id: &str, metric: &PerformanceMetric) -> SklResult<()> {
        let _lock = self.lock.write().unwrap();

        // Validate metric
        if !metric.is_valid() {
            return Err(SklearsError::InvalidInput("Invalid metric".to_string()));
        }

        // Check storage limits
        self.check_limits(session_id)?;

        // Store metric
        let metrics = self.storage.entry(session_id.to_string()).or_insert_with(Vec::new);
        metrics.push(metric.clone());

        // Update statistics
        self.stats.total_metrics_stored += 1;
        self.stats.last_update = SystemTime::now();

        Ok(())
    }

    fn retrieve_metrics(&self, session_id: &str, time_range: &TimeRange) -> SklResult<Vec<PerformanceMetric>> {
        let _lock = self.lock.read().unwrap();

        if let Some(metrics) = self.storage.get(session_id) {
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

    fn aggregate_metrics(&self, session_id: &str, aggregation: &MetricsAggregationConfig) -> SklResult<Vec<PerformanceMetric>> {
        let _lock = self.lock.read().unwrap();

        if let Some(metrics) = self.storage.get(session_id) {
            let aggregator = MetricsAggregator::new(aggregation.clone());
            aggregator.aggregate(metrics)
        } else {
            Ok(Vec::new())
        }
    }

    fn metrics_count(&self, session_id: &str) -> SklResult<u64> {
        let _lock = self.lock.read().unwrap();

        let count = self.storage.get(session_id).map_or(0, |metrics| metrics.len()) as u64;
        Ok(count)
    }

    fn delete_metrics(&mut self, session_id: &str) -> SklResult<u64> {
        let _lock = self.lock.write().unwrap();

        let count = self.storage.remove(session_id).map_or(0, |metrics| metrics.len()) as u64;
        self.stats.total_metrics_deleted += count;
        Ok(count)
    }

    fn storage_stats(&self) -> StorageStatistics {
        self.stats.clone()
    }

    fn cleanup_old_metrics(&mut self, older_than: Duration) -> SklResult<u64> {
        let _lock = self.lock.write().unwrap();
        let cutoff_time = SystemTime::now() - older_than;
        let mut removed_count = 0;

        for metrics in self.storage.values_mut() {
            let original_len = metrics.len();
            metrics.retain(|metric| metric.timestamp >= cutoff_time);
            removed_count += original_len - metrics.len();
        }

        self.stats.total_metrics_deleted += removed_count as u64;
        Ok(removed_count as u64)
    }
}

impl Default for InMemoryMetricsStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Maximum metrics per session
    pub max_metrics_per_session: usize,

    /// Maximum total metrics across all sessions
    pub max_total_metrics: usize,

    /// Enable compression
    pub enable_compression: bool,

    /// Compression threshold (number of metrics)
    pub compression_threshold: usize,

    /// Retention period
    pub retention_period: Duration,

    /// Enable automatic cleanup
    pub auto_cleanup: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_metrics_per_session: 100_000,
            max_total_metrics: 1_000_000,
            enable_compression: false,
            compression_threshold: 10_000,
            retention_period: Duration::from_secs(24 * 3600), // 24 hours
            auto_cleanup: true,
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    /// Total metrics stored
    pub total_metrics_stored: u64,

    /// Total metrics deleted
    pub total_metrics_deleted: u64,

    /// Storage size (bytes)
    pub storage_size_bytes: u64,

    /// Number of sessions
    pub session_count: usize,

    /// Last update timestamp
    pub last_update: SystemTime,

    /// Storage efficiency (0.0 to 1.0)
    pub efficiency: f64,
}

impl StorageStatistics {
    fn new() -> Self {
        Self {
            total_metrics_stored: 0,
            total_metrics_deleted: 0,
            storage_size_bytes: 0,
            session_count: 0,
            last_update: SystemTime::now(),
            efficiency: 1.0,
        }
    }
}

/// Metrics aggregation configuration
#[derive(Debug, Clone)]
pub struct MetricsAggregationConfig {
    /// Time window for aggregation
    pub time_window: Duration,

    /// Aggregation types to apply
    pub aggregation_types: Vec<AggregationType>,

    /// Metrics to include (empty = all)
    pub metric_names: Vec<String>,

    /// Tag filters
    pub tag_filters: HashMap<String, String>,

    /// Group by tags
    pub group_by_tags: Vec<String>,

    /// Maximum number of groups
    pub max_groups: usize,
}

impl Default for MetricsAggregationConfig {
    fn default() -> Self {
        Self {
            time_window: Duration::from_secs(60),
            aggregation_types: vec![AggregationType::Average, AggregationType::Maximum],
            metric_names: Vec::new(),
            tag_filters: HashMap::new(),
            group_by_tags: Vec::new(),
            max_groups: 100,
        }
    }
}

/// Types of aggregation operations
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationType {
    /// Average value
    Average,

    /// Minimum value
    Minimum,

    /// Maximum value
    Maximum,

    /// Sum of values
    Sum,

    /// Count of values
    Count,

    /// Standard deviation
    StandardDeviation,

    /// Percentile calculation
    Percentile { percentile: f64 },

    /// Rate calculation (per second)
    Rate,

    /// First value in window
    First,

    /// Last value in window
    Last,
}

/// Metrics aggregator for calculating aggregate values
#[derive(Debug)]
pub struct MetricsAggregator {
    config: MetricsAggregationConfig,
}

impl MetricsAggregator {
    /// Create new aggregator
    pub fn new(config: MetricsAggregationConfig) -> Self {
        Self { config }
    }

    /// Aggregate metrics according to configuration
    pub fn aggregate(&self, metrics: &[PerformanceMetric]) -> SklResult<Vec<PerformanceMetric>> {
        let mut result = Vec::new();

        // Filter metrics by name if specified
        let filtered_metrics: Vec<&PerformanceMetric> = if self.config.metric_names.is_empty() {
            metrics.iter().collect()
        } else {
            metrics.iter()
                .filter(|m| self.config.metric_names.contains(&m.name))
                .collect()
        };

        // Apply tag filters
        let tag_filtered_metrics: Vec<&PerformanceMetric> = filtered_metrics.iter()
            .filter(|m| m.matches_tags(&self.config.tag_filters))
            .cloned()
            .collect();

        // Group metrics
        let groups = self.group_metrics(&tag_filtered_metrics);

        // Aggregate each group
        for (group_key, group_metrics) in groups {
            for aggregation_type in &self.config.aggregation_types {
                if let Some(aggregated) = self.aggregate_group(&group_metrics, aggregation_type, &group_key)? {
                    result.push(aggregated);
                }
            }
        }

        Ok(result)
    }

    /// Group metrics by specified tags
    fn group_metrics(&self, metrics: &[&PerformanceMetric]) -> HashMap<String, Vec<&PerformanceMetric>> {
        let mut groups: HashMap<String, Vec<&PerformanceMetric>> = HashMap::new();

        for metric in metrics {
            let group_key = if self.config.group_by_tags.is_empty() {
                "default".to_string()
            } else {
                let key_parts: Vec<String> = self.config.group_by_tags.iter()
                    .map(|tag| {
                        let value = metric.tags.get(tag).unwrap_or(&"unknown".to_string());
                        format!("{}={}", tag, value)
                    })
                    .collect();
                key_parts.join(",")
            };

            groups.entry(group_key).or_insert_with(Vec::new).push(metric);
        }

        groups
    }

    /// Aggregate a group of metrics
    fn aggregate_group(&self, metrics: &[&PerformanceMetric], aggregation_type: &AggregationType, group_key: &str) -> SklResult<Option<PerformanceMetric>> {
        if metrics.is_empty() {
            return Ok(None);
        }

        let values: Vec<f64> = metrics.iter().map(|m| m.value).collect();
        let timestamps: Vec<SystemTime> = metrics.iter().map(|m| m.timestamp).collect();

        let (aggregated_value, metric_name_suffix) = match aggregation_type {
            AggregationType::Average => {
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                (avg, "_avg")
            }
            AggregationType::Minimum => {
                let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                (min, "_min")
            }
            AggregationType::Maximum => {
                let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                (max, "_max")
            }
            AggregationType::Sum => {
                let sum = values.iter().sum::<f64>();
                (sum, "_sum")
            }
            AggregationType::Count => {
                let count = values.len() as f64;
                (count, "_count")
            }
            AggregationType::StandardDeviation => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                let std_dev = variance.sqrt();
                (std_dev, "_stddev")
            }
            AggregationType::Percentile { percentile } => {
                let mut sorted_values = values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let index = ((*percentile / 100.0) * (sorted_values.len() - 1) as f64) as usize;
                let value = sorted_values[index.min(sorted_values.len() - 1)];
                (value, &format!("_p{}", *percentile as u32))
            }
            AggregationType::Rate => {
                if timestamps.len() < 2 {
                    return Ok(None);
                }
                let time_span = timestamps.iter().max().unwrap()
                    .duration_since(*timestamps.iter().min().unwrap())
                    .unwrap_or(Duration::from_secs(1));
                let rate = values.len() as f64 / time_span.as_secs_f64();
                (rate, "_rate")
            }
            AggregationType::First => {
                (values[0], "_first")
            }
            AggregationType::Last => {
                (values[values.len() - 1], "_last")
            }
        };

        // Create aggregated metric
        let base_metric = metrics[0];
        let mut aggregated_metric = PerformanceMetric {
            name: format!("{}{}", base_metric.name, metric_name_suffix),
            value: aggregated_value,
            unit: base_metric.unit.clone(),
            timestamp: SystemTime::now(),
            tags: base_metric.tags.clone(),
            metric_type: MetricType::Gauge, // Aggregated metrics are typically gauges
            source: MetricSource::Derived,
            quality: MetricQuality::default(),
            metadata: HashMap::new(),
        };

        // Add aggregation metadata
        aggregated_metric.metadata.insert("aggregation_type".to_string(), format!("{:?}", aggregation_type));
        aggregated_metric.metadata.insert("group_key".to_string(), group_key.to_string());
        aggregated_metric.metadata.insert("sample_count".to_string(), metrics.len().to_string());

        Ok(Some(aggregated_metric))
    }
}

/// Metrics collector for automated collection
#[derive(Debug)]
pub struct MetricsCollector {
    /// Collection configuration
    config: MetricsCollectionConfig,

    /// Collection interval
    collection_interval: Duration,

    /// Registered collectors
    collectors: Vec<Box<dyn MetricCollectorFunction>>,

    /// Collection statistics
    stats: CollectionStatistics,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(config: MetricsCollectionConfig) -> Self {
        Self {
            collection_interval: config.collection_interval,
            config,
            collectors: Vec::new(),
            stats: CollectionStatistics::new(),
        }
    }

    /// Register a collector function
    pub fn register_collector(&mut self, collector: Box<dyn MetricCollectorFunction>) {
        self.collectors.push(collector);
    }

    /// Collect all metrics
    pub fn collect_metrics(&mut self) -> SklResult<Vec<PerformanceMetric>> {
        let mut metrics = Vec::new();
        let start_time = Instant::now();

        for collector in &self.collectors {
            match collector.collect() {
                Ok(mut collected) => {
                    metrics.append(&mut collected);
                    self.stats.successful_collections += 1;
                }
                Err(e) => {
                    self.stats.failed_collections += 1;
                    log::warn!("Metric collection failed: {}", e);
                }
            }
        }

        let collection_time = start_time.elapsed();
        self.stats.total_collection_time += collection_time;
        self.stats.last_collection = SystemTime::now();

        Ok(metrics)
    }

    /// Get collection statistics
    pub fn statistics(&self) -> &CollectionStatistics {
        &self.stats
    }
}

/// Trait for metric collector functions
pub trait MetricCollectorFunction: Send + Sync {
    /// Collect metrics
    fn collect(&self) -> SklResult<Vec<PerformanceMetric>>;

    /// Get collector name
    fn name(&self) -> &str;

    /// Get collector description
    fn description(&self) -> &str;
}

/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStatistics {
    /// Number of successful collections
    pub successful_collections: u64,

    /// Number of failed collections
    pub failed_collections: u64,

    /// Total collection time
    pub total_collection_time: Duration,

    /// Last collection timestamp
    pub last_collection: SystemTime,

    /// Average collection time
    pub avg_collection_time: Duration,
}

impl CollectionStatistics {
    fn new() -> Self {
        Self {
            successful_collections: 0,
            failed_collections: 0,
            total_collection_time: Duration::from_millis(0),
            last_collection: SystemTime::now(),
            avg_collection_time: Duration::from_millis(0),
        }
    }

    /// Calculate collection success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.successful_collections + self.failed_collections;
        if total > 0 {
            self.successful_collections as f64 / total as f64
        } else {
            1.0
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metric_creation() {
        let metric = PerformanceMetric::new(
            "cpu_usage".to_string(),
            0.75,
            "percentage".to_string(),
        );

        assert_eq!(metric.name, "cpu_usage");
        assert_eq!(metric.value, 0.75);
        assert_eq!(metric.unit, "percentage");
        assert!(metric.is_valid());
    }

    #[test]
    fn test_metric_builder() {
        let metric = PerformanceMetric::builder()
            .name("memory_usage".to_string())
            .value(1024.0)
            .unit("MB".to_string())
            .tag("host".to_string(), "server1".to_string())
            .metric_type(MetricType::Gauge)
            .build()
            .unwrap();

        assert_eq!(metric.name, "memory_usage");
        assert_eq!(metric.value, 1024.0);
        assert!(metric.tags.contains_key("host"));
        assert_eq!(metric.metric_type, MetricType::Gauge);
    }

    #[test]
    fn test_in_memory_storage() {
        let mut storage = InMemoryMetricsStorage::new();
        let metric = PerformanceMetric::new(
            "test_metric".to_string(),
            42.0,
            "value".to_string(),
        );

        // Store metric
        storage.store_metric("session1", &metric).unwrap();

        // Check count
        let count = storage.metrics_count("session1").unwrap();
        assert_eq!(count, 1);

        // Retrieve metrics
        let time_range = TimeRange::new(
            SystemTime::now() - Duration::from_secs(60),
            SystemTime::now() + Duration::from_secs(60),
        );
        let metrics = storage.retrieve_metrics("session1", &time_range).unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "test_metric");
    }

    #[test]
    fn test_metrics_aggregation() {
        let metrics = vec![
            PerformanceMetric::new("cpu".to_string(), 10.0, "%".to_string()),
            PerformanceMetric::new("cpu".to_string(), 20.0, "%".to_string()),
            PerformanceMetric::new("cpu".to_string(), 30.0, "%".to_string()),
        ];

        let config = MetricsAggregationConfig {
            aggregation_types: vec![AggregationType::Average, AggregationType::Maximum],
            ..Default::default()
        };

        let aggregator = MetricsAggregator::new(config);
        let aggregated = aggregator.aggregate(&metrics).unwrap();

        assert_eq!(aggregated.len(), 2); // Average and Maximum

        // Find average metric
        let avg_metric = aggregated.iter().find(|m| m.name.contains("_avg")).unwrap();
        assert_eq!(avg_metric.value, 20.0);

        // Find max metric
        let max_metric = aggregated.iter().find(|m| m.name.contains("_max")).unwrap();
        assert_eq!(max_metric.value, 30.0);
    }

    #[test]
    fn test_metric_quality() {
        let mut quality = MetricQuality::default();
        assert_eq!(quality.overall_score(), 1.0);

        quality.accuracy = 0.8;
        quality.completeness = 0.9;
        assert_eq!(quality.overall_score(), 0.925);
        assert!(quality.meets_threshold(0.9));
        assert!(!quality.meets_threshold(0.95));
    }

    #[test]
    fn test_metric_types() {
        assert!(MetricType::Counter.is_cumulative());
        assert!(!MetricType::Gauge.is_cumulative());

        assert!(MetricType::Rate.is_rate());
        assert!(MetricType::Throughput.is_rate());

        assert!(MetricType::Timer.is_duration());
        assert!(MetricType::Latency.is_duration());

        assert_eq!(MetricType::ExecutionTime.default_unit(), "milliseconds");
        assert_eq!(MetricType::MemoryUsage.default_unit(), "bytes");
    }

    #[test]
    fn test_storage_limits() {
        let config = StorageConfig {
            max_metrics_per_session: 2,
            max_total_metrics: 5,
            ..Default::default()
        };

        let mut storage = InMemoryMetricsStorage::with_config(config);
        let metric = PerformanceMetric::new("test".to_string(), 1.0, "value".to_string());

        // Should succeed for first two metrics
        storage.store_metric("session1", &metric).unwrap();
        storage.store_metric("session1", &metric).unwrap();

        // Should fail for third metric (exceeds per-session limit)
        let result = storage.store_metric("session1", &metric);
        assert!(result.is_err());
    }
}