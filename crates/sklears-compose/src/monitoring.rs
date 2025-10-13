//! Pipeline monitoring and performance profiling utilities
//!
//! This module provides comprehensive monitoring, profiling, and observability
//! tools for pipeline execution, including real-time metrics collection,
//! performance analysis, and anomaly detection.

use sklears_core::{traits::Estimator, types::FloatBounds};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Pipeline execution monitor with real-time metrics collection
pub struct PipelineMonitor {
    /// Monitor configuration
    config: MonitorConfig,
    /// Execution metrics storage
    metrics: Arc<Mutex<MetricsStorage>>,
    /// Active execution contexts
    active_contexts: Arc<Mutex<HashMap<String, ExecutionContext>>>,
    /// Performance baselines
    baselines: HashMap<String, PerformanceBaseline>,
}

impl PipelineMonitor {
    /// Create a new pipeline monitor
    #[must_use]
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(Mutex::new(MetricsStorage::new())),
            active_contexts: Arc::new(Mutex::new(HashMap::new())),
            baselines: HashMap::new(),
        }
    }

    /// Start monitoring a pipeline execution
    #[must_use]
    pub fn start_execution(&self, execution_id: &str, pipeline_name: &str) -> ExecutionHandle {
        let context = ExecutionContext::new(execution_id, pipeline_name);

        if let Ok(mut contexts) = self.active_contexts.lock() {
            contexts.insert(execution_id.to_string(), context);
        }

        ExecutionHandle::new(execution_id.to_string(), self.metrics.clone())
    }

    /// Record a metric
    pub fn record_metric(&self, metric: Metric) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.add_metric(metric);
        }
    }

    /// Get current metrics snapshot
    #[must_use]
    pub fn get_metrics_snapshot(&self) -> MetricsSnapshot {
        if let Ok(metrics) = self.metrics.lock() {
            metrics.snapshot()
        } else {
            MetricsSnapshot::empty()
        }
    }

    /// Analyze performance trends
    #[must_use]
    pub fn analyze_performance(&self, pipeline_name: &str) -> PerformanceAnalysis {
        let snapshot = self.get_metrics_snapshot();
        let pipeline_metrics = snapshot.filter_by_pipeline(pipeline_name);

        PerformanceAnalysis::from_metrics(pipeline_metrics)
    }

    /// Detect anomalies in pipeline execution
    #[must_use]
    pub fn detect_anomalies(&self, pipeline_name: &str) -> Vec<Anomaly> {
        let snapshot = self.get_metrics_snapshot();
        let pipeline_metrics = snapshot.filter_by_pipeline(pipeline_name);

        let mut anomalies = Vec::new();

        // Check for execution time anomalies
        if let Some(baseline) = self.baselines.get(pipeline_name) {
            for metric in &pipeline_metrics.execution_times {
                if metric.value > baseline.avg_execution_time * 2.0 {
                    anomalies.push(Anomaly {
                        anomaly_type: AnomalyType::SlowExecution,
                        severity: AnomalySeverity::High,
                        timestamp: metric.timestamp,
                        description: format!(
                            "Execution time {:.2}s is significantly higher than baseline {:.2}s",
                            metric.value, baseline.avg_execution_time
                        ),
                        pipeline_name: pipeline_name.to_string(),
                        metric_name: metric.name.clone(),
                    });
                }
            }
        }

        // Check for memory usage anomalies
        for metric in &pipeline_metrics.memory_usage {
            if metric.value > self.config.memory_threshold_mb {
                anomalies.push(Anomaly {
                    anomaly_type: AnomalyType::HighMemoryUsage,
                    severity: AnomalySeverity::Medium,
                    timestamp: metric.timestamp,
                    description: format!(
                        "Memory usage {:.2}MB exceeds threshold {:.2}MB",
                        metric.value, self.config.memory_threshold_mb
                    ),
                    pipeline_name: pipeline_name.to_string(),
                    metric_name: metric.name.clone(),
                });
            }
        }

        anomalies
    }

    /// Set performance baseline for a pipeline
    pub fn set_baseline(&mut self, pipeline_name: &str, baseline: PerformanceBaseline) {
        self.baselines.insert(pipeline_name.to_string(), baseline);
    }

    /// Get active execution contexts
    #[must_use]
    pub fn get_active_executions(&self) -> Vec<ExecutionContext> {
        if let Ok(contexts) = self.active_contexts.lock() {
            contexts.values().cloned().collect()
        } else {
            Vec::new()
        }
    }
}

/// Monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Maximum number of metrics to retain
    pub max_metrics: usize,
    /// Sampling interval for continuous metrics
    pub sampling_interval: Duration,
    /// Memory usage threshold for anomaly detection (MB)
    pub memory_threshold_mb: f64,
    /// Execution time threshold for anomaly detection (seconds)
    pub execution_time_threshold_sec: f64,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Enable distributed tracing
    pub enable_tracing: bool,
}

impl MonitorConfig {
    /// Create a new monitor configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_metrics: 10000,
            sampling_interval: Duration::from_secs(1),
            memory_threshold_mb: 1024.0,        // 1GB
            execution_time_threshold_sec: 60.0, // 1 minute
            enable_profiling: true,
            enable_tracing: false,
        }
    }

    /// Set maximum metrics retention
    #[must_use]
    pub fn max_metrics(mut self, max: usize) -> Self {
        self.max_metrics = max;
        self
    }

    /// Set sampling interval
    #[must_use]
    pub fn sampling_interval(mut self, interval: Duration) -> Self {
        self.sampling_interval = interval;
        self
    }

    /// Set memory threshold
    #[must_use]
    pub fn memory_threshold_mb(mut self, threshold: f64) -> Self {
        self.memory_threshold_mb = threshold;
        self
    }

    /// Enable profiling
    #[must_use]
    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    /// Enable tracing
    #[must_use]
    pub fn enable_tracing(mut self, enable: bool) -> Self {
        self.enable_tracing = enable;
        self
    }
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution handle for tracking a specific pipeline run
pub struct ExecutionHandle {
    execution_id: String,
    start_time: Instant,
    metrics: Arc<Mutex<MetricsStorage>>,
    stage_timings: HashMap<String, Instant>,
}

impl ExecutionHandle {
    /// Create a new execution handle
    fn new(execution_id: String, metrics: Arc<Mutex<MetricsStorage>>) -> Self {
        Self {
            execution_id,
            start_time: Instant::now(),
            metrics,
            stage_timings: HashMap::new(),
        }
    }

    /// Start timing a pipeline stage
    pub fn start_stage(&mut self, stage_name: &str) {
        self.stage_timings
            .insert(stage_name.to_string(), Instant::now());
    }

    /// End timing a pipeline stage
    pub fn end_stage(&mut self, stage_name: &str) {
        if let Some(start_time) = self.stage_timings.remove(stage_name) {
            let duration = start_time.elapsed();

            let metric = Metric {
                name: format!("stage_duration_{stage_name}"),
                value: duration.as_secs_f64(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                pipeline_name: self.execution_id.clone(),
                stage_name: Some(stage_name.to_string()),
                execution_id: Some(self.execution_id.clone()),
                metadata: HashMap::new(),
            };

            if let Ok(mut metrics) = self.metrics.lock() {
                metrics.add_metric(metric);
            }
        }
    }

    /// Record custom metric
    pub fn record_metric(&self, name: &str, value: f64) {
        let metric = Metric {
            name: name.to_string(),
            value,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            pipeline_name: self.execution_id.clone(),
            stage_name: None,
            execution_id: Some(self.execution_id.clone()),
            metadata: HashMap::new(),
        };

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.add_metric(metric);
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, usage_mb: f64) {
        self.record_metric("memory_usage_mb", usage_mb);
    }

    /// Record throughput
    pub fn record_throughput(&self, samples_per_sec: f64) {
        self.record_metric("throughput_samples_per_sec", samples_per_sec);
    }

    /// Finish execution and record total time
    pub fn finish(self) {
        let total_duration = self.start_time.elapsed();

        let metric = Metric {
            name: "total_execution_time".to_string(),
            value: total_duration.as_secs_f64(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            pipeline_name: self.execution_id.clone(),
            stage_name: None,
            execution_id: Some(self.execution_id.clone()),
            metadata: HashMap::new(),
        };

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.add_metric(metric);
        }
    }
}

/// Execution context for active pipeline runs
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Unique execution identifier
    pub execution_id: String,
    /// Pipeline name
    pub pipeline_name: String,
    /// Start timestamp
    pub start_time: u64,
    /// Current stage
    pub current_stage: Option<String>,
    /// Execution status
    pub status: ExecutionStatus,
}

impl ExecutionContext {
    /// Create a new execution context
    fn new(execution_id: &str, pipeline_name: &str) -> Self {
        Self {
            execution_id: execution_id.to_string(),
            pipeline_name: pipeline_name.to_string(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            current_stage: None,
            status: ExecutionStatus::Running,
        }
    }
}

/// Execution status
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    /// Pipeline is currently running
    Running,
    /// Pipeline completed successfully
    Completed,
    /// Pipeline failed with error
    Failed,
    /// Pipeline was cancelled
    Cancelled,
}

/// Individual metric measurement
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// Pipeline name
    pub pipeline_name: String,
    /// Optional stage name
    pub stage_name: Option<String>,
    /// Optional execution ID
    pub execution_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Metrics storage with circular buffer
struct MetricsStorage {
    metrics: VecDeque<Metric>,
    max_size: usize,
}

impl MetricsStorage {
    /// Create new metrics storage
    fn new() -> Self {
        Self {
            metrics: VecDeque::new(),
            max_size: 10000,
        }
    }

    /// Add a metric
    fn add_metric(&mut self, metric: Metric) {
        if self.metrics.len() >= self.max_size {
            self.metrics.pop_front();
        }
        self.metrics.push_back(metric);
    }

    /// Create a snapshot of current metrics
    fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            execution_times: self
                .metrics
                .iter()
                .filter(|m| m.name.contains("execution_time"))
                .cloned()
                .collect(),
            memory_usage: self
                .metrics
                .iter()
                .filter(|m| m.name.contains("memory_usage"))
                .cloned()
                .collect(),
            throughput: self
                .metrics
                .iter()
                .filter(|m| m.name.contains("throughput"))
                .cloned()
                .collect(),
            stage_durations: self
                .metrics
                .iter()
                .filter(|m| m.name.starts_with("stage_duration"))
                .cloned()
                .collect(),
            custom_metrics: self
                .metrics
                .iter()
                .filter(|m| {
                    !m.name.contains("execution_time")
                        && !m.name.contains("memory_usage")
                        && !m.name.contains("throughput")
                        && !m.name.starts_with("stage_duration")
                })
                .cloned()
                .collect(),
        }
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Execution time metrics
    pub execution_times: Vec<Metric>,
    /// Memory usage metrics
    pub memory_usage: Vec<Metric>,
    /// Throughput metrics
    pub throughput: Vec<Metric>,
    /// Stage duration metrics
    pub stage_durations: Vec<Metric>,
    /// Custom metrics
    pub custom_metrics: Vec<Metric>,
}

impl MetricsSnapshot {
    /// Create an empty snapshot
    fn empty() -> Self {
        Self {
            execution_times: Vec::new(),
            memory_usage: Vec::new(),
            throughput: Vec::new(),
            stage_durations: Vec::new(),
            custom_metrics: Vec::new(),
        }
    }

    /// Filter metrics by pipeline name
    fn filter_by_pipeline(&self, pipeline_name: &str) -> Self {
        Self {
            execution_times: self
                .execution_times
                .iter()
                .filter(|m| m.pipeline_name == pipeline_name)
                .cloned()
                .collect(),
            memory_usage: self
                .memory_usage
                .iter()
                .filter(|m| m.pipeline_name == pipeline_name)
                .cloned()
                .collect(),
            throughput: self
                .throughput
                .iter()
                .filter(|m| m.pipeline_name == pipeline_name)
                .cloned()
                .collect(),
            stage_durations: self
                .stage_durations
                .iter()
                .filter(|m| m.pipeline_name == pipeline_name)
                .cloned()
                .collect(),
            custom_metrics: self
                .custom_metrics
                .iter()
                .filter(|m| m.pipeline_name == pipeline_name)
                .cloned()
                .collect(),
        }
    }

    /// Get all metrics as a single vector
    #[must_use]
    pub fn all_metrics(&self) -> Vec<&Metric> {
        let mut all = Vec::new();
        all.extend(&self.execution_times);
        all.extend(&self.memory_usage);
        all.extend(&self.throughput);
        all.extend(&self.stage_durations);
        all.extend(&self.custom_metrics);
        all
    }
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Average execution time
    pub avg_execution_time: f64,
    /// Execution time percentiles
    pub execution_time_percentiles: HashMap<u8, f64>,
    /// Average memory usage
    pub avg_memory_usage: f64,
    /// Peak memory usage
    pub peak_memory_usage: f64,
    /// Average throughput
    pub avg_throughput: f64,
    /// Stage performance breakdown
    pub stage_breakdown: HashMap<String, StagePerformance>,
    /// Performance trends
    pub trends: PerformanceTrends,
}

impl PerformanceAnalysis {
    /// Create performance analysis from metrics
    fn from_metrics(metrics: MetricsSnapshot) -> Self {
        let mut execution_times: Vec<f64> =
            metrics.execution_times.iter().map(|m| m.value).collect();
        execution_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_execution_time = if execution_times.is_empty() {
            0.0
        } else {
            execution_times.iter().sum::<f64>() / execution_times.len() as f64
        };

        let execution_time_percentiles = Self::calculate_percentiles(&execution_times);

        let memory_values: Vec<f64> = metrics.memory_usage.iter().map(|m| m.value).collect();
        let avg_memory_usage = if memory_values.is_empty() {
            0.0
        } else {
            memory_values.iter().sum::<f64>() / memory_values.len() as f64
        };
        let peak_memory_usage = memory_values.iter().fold(0.0f64, |acc, &x| acc.max(x));

        let throughput_values: Vec<f64> = metrics.throughput.iter().map(|m| m.value).collect();
        let avg_throughput = if throughput_values.is_empty() {
            0.0
        } else {
            throughput_values.iter().sum::<f64>() / throughput_values.len() as f64
        };

        let stage_breakdown = Self::analyze_stages(&metrics.stage_durations);
        let trends = Self::analyze_trends(&metrics);

        Self {
            avg_execution_time,
            execution_time_percentiles,
            avg_memory_usage,
            peak_memory_usage,
            avg_throughput,
            stage_breakdown,
            trends,
        }
    }

    /// Calculate percentiles for a sorted vector
    fn calculate_percentiles(sorted_values: &[f64]) -> HashMap<u8, f64> {
        let mut percentiles = HashMap::new();

        if sorted_values.is_empty() {
            return percentiles;
        }

        for p in &[50, 75, 90, 95, 99] {
            let index =
                ((f64::from(*p) / 100.0) * (sorted_values.len() - 1) as f64).round() as usize;
            let index = index.min(sorted_values.len() - 1);
            percentiles.insert(*p, sorted_values[index]);
        }

        percentiles
    }

    /// Analyze stage performance
    fn analyze_stages(stage_metrics: &[Metric]) -> HashMap<String, StagePerformance> {
        let mut stage_map: HashMap<String, Vec<f64>> = HashMap::new();

        for metric in stage_metrics {
            if let Some(stage_name) = &metric.stage_name {
                stage_map
                    .entry(stage_name.clone())
                    .or_default()
                    .push(metric.value);
            }
        }

        let mut stage_breakdown = HashMap::new();
        for (stage_name, times) in stage_map {
            let avg_time = times.iter().sum::<f64>() / times.len() as f64;
            let min_time = times.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
            let max_time = times.iter().fold(0.0f64, |acc, &x| acc.max(x));

            stage_breakdown.insert(
                stage_name,
                StagePerformance {
                    avg_duration: avg_time,
                    min_duration: min_time,
                    max_duration: max_time,
                    execution_count: times.len(),
                },
            );
        }

        stage_breakdown
    }

    /// Analyze performance trends
    fn analyze_trends(metrics: &MetricsSnapshot) -> PerformanceTrends {
        // Simple trend analysis - in practice this would be more sophisticated
        let recent_count = 10;

        let recent_execution_times: Vec<f64> = metrics
            .execution_times
            .iter()
            .rev()
            .take(recent_count)
            .map(|m| m.value)
            .collect();

        let execution_time_trend = if recent_execution_times.len() >= 2 {
            let first_half = &recent_execution_times[recent_execution_times.len() / 2..];
            let second_half = &recent_execution_times[..recent_execution_times.len() / 2];

            let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
            let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;

            if second_avg > first_avg * 1.1 {
                Trend::Increasing
            } else if second_avg < first_avg * 0.9 {
                Trend::Decreasing
            } else {
                Trend::Stable
            }
        } else {
            Trend::Unknown
        };

        PerformanceTrends {
            execution_time_trend,
            memory_usage_trend: Trend::Unknown, // Would implement similar logic
            throughput_trend: Trend::Unknown,
        }
    }
}

/// Stage performance metrics
#[derive(Debug, Clone)]
pub struct StagePerformance {
    /// Average duration
    pub avg_duration: f64,
    /// Minimum duration
    pub min_duration: f64,
    /// Maximum duration
    pub max_duration: f64,
    /// Number of executions
    pub execution_count: usize,
}

/// Performance trends
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Execution time trend
    pub execution_time_trend: Trend,
    /// Memory usage trend
    pub memory_usage_trend: Trend,
    /// Throughput trend
    pub throughput_trend: Trend,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum Trend {
    /// Metric is increasing
    Increasing,
    /// Metric is decreasing
    Decreasing,
    /// Metric is stable
    Stable,
    /// Trend is unknown
    Unknown,
}

/// Performance baseline for anomaly detection
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Average execution time
    pub avg_execution_time: f64,
    /// Standard deviation of execution time
    pub std_dev_execution_time: f64,
    /// Average memory usage
    pub avg_memory_usage: f64,
    /// Average throughput
    pub avg_throughput: f64,
}

/// Detected anomaly
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Timestamp when detected
    pub timestamp: u64,
    /// Human-readable description
    pub description: String,
    /// Pipeline name
    pub pipeline_name: String,
    /// Metric name
    pub metric_name: String,
}

/// Types of anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// Execution time is unusually slow
    SlowExecution,
    /// Memory usage is unusually high
    HighMemoryUsage,
    /// Throughput is unusually low
    LowThroughput,
    /// Error rate is unusually high
    HighErrorRate,
    /// Resource contention detected
    ResourceContention,
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalySeverity {
    /// Low severity - informational
    Low,
    /// Medium severity - worth investigating
    Medium,
    /// High severity - requires immediate attention
    High,
    /// Critical severity - system may be failing
    Critical,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_config() {
        let config = MonitorConfig::new()
            .max_metrics(5000)
            .memory_threshold_mb(512.0)
            .enable_profiling(true);

        assert_eq!(config.max_metrics, 5000);
        assert_eq!(config.memory_threshold_mb, 512.0);
        assert!(config.enable_profiling);
    }

    #[test]
    fn test_pipeline_monitor() {
        let config = MonitorConfig::new();
        let monitor = PipelineMonitor::new(config);

        let mut handle = monitor.start_execution("test-exec-1", "test-pipeline");
        handle.start_stage("preprocessing");
        handle.end_stage("preprocessing");
        handle.record_memory_usage(256.0);
        handle.finish();

        let snapshot = monitor.get_metrics_snapshot();
        assert!(!snapshot.stage_durations.is_empty());
        assert!(!snapshot.memory_usage.is_empty());
    }

    #[test]
    fn test_execution_handle() {
        let metrics = Arc::new(Mutex::new(MetricsStorage::new()));
        let mut handle = ExecutionHandle::new("test-id".to_string(), metrics.clone());

        handle.start_stage("test-stage");
        std::thread::sleep(Duration::from_millis(10));
        handle.end_stage("test-stage");

        let metrics_lock = metrics.lock().unwrap();
        assert!(!metrics_lock.metrics.is_empty());
    }

    #[test]
    fn test_metrics_snapshot() {
        let mut snapshot = MetricsSnapshot::empty();
        let metric = Metric {
            name: "test_metric".to_string(),
            value: 42.0,
            timestamp: 123456789,
            pipeline_name: "test-pipeline".to_string(),
            stage_name: None,
            execution_id: None,
            metadata: HashMap::new(),
        };

        snapshot.custom_metrics.push(metric);
        assert_eq!(snapshot.all_metrics().len(), 1);
    }

    #[test]
    fn test_performance_analysis() {
        let mut snapshot = MetricsSnapshot::empty();

        // Add some execution time metrics
        for i in 1..=5 {
            snapshot.execution_times.push(Metric {
                name: "execution_time".to_string(),
                value: i as f64,
                timestamp: 123456789,
                pipeline_name: "test".to_string(),
                stage_name: None,
                execution_id: None,
                metadata: HashMap::new(),
            });
        }

        let analysis = PerformanceAnalysis::from_metrics(snapshot);
        assert_eq!(analysis.avg_execution_time, 3.0);
        assert!(analysis.execution_time_percentiles.contains_key(&50));
    }

    #[test]
    fn test_anomaly_detection() {
        let config = MonitorConfig::new().memory_threshold_mb(100.0);
        let mut monitor = PipelineMonitor::new(config);

        let baseline = PerformanceBaseline {
            avg_execution_time: 1.0,
            std_dev_execution_time: 0.1,
            avg_memory_usage: 50.0,
            avg_throughput: 1000.0,
        };
        monitor.set_baseline("test-pipeline", baseline);

        // Record a metric that should trigger anomaly
        monitor.record_metric(Metric {
            name: "memory_usage_mb".to_string(),
            value: 200.0, // Above threshold
            timestamp: 123456789,
            pipeline_name: "test-pipeline".to_string(),
            stage_name: None,
            execution_id: None,
            metadata: HashMap::new(),
        });

        let anomalies = monitor.detect_anomalies("test-pipeline");
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::HighMemoryUsage);
    }
}
