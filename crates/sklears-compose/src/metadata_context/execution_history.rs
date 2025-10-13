//! # Execution History Module
//!
//! Comprehensive execution tracking and history management system for monitoring
//! algorithm performance, resource usage, and execution patterns over time.
//!
//! ## Features
//!
//! - **Execution Tracking**: Record detailed execution information
//! - **Performance History**: Track metrics and trends over time
//! - **Resource Monitoring**: Monitor CPU, memory, and I/O usage
//! - **Error Analysis**: Comprehensive failure tracking and analysis
//! - **Timeline Management**: Organize executions chronologically
//! - **Comparison Tools**: Compare performance across executions
//! - **Analytics Engine**: Generate insights from execution patterns
//! - **Alerting System**: Monitor for anomalies and performance degradation
//!
//! ## Architecture
//!
//! ```text
//! ExecutionHistory
//! ├── ExecutionTracker (core tracking functionality)
//! ├── PerformanceAnalyzer (metrics analysis and trends)
//! ├── ResourceMonitor (resource usage tracking)
//! ├── ErrorTracker (failure analysis and categorization)
//! ├── TimelineManager (chronological organization)
//! ├── ComparisonEngine (performance comparison tools)
//! └── AlertingSystem (anomaly detection and notifications)
//! ```

use scirs2_core::error::{CoreError, Result};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::ndarray::{Array, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Comprehensive execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Unique execution identifier
    pub execution_id: String,
    /// Algorithm or process name
    pub algorithm_name: String,
    /// Execution start time
    pub start_time: SystemTime,
    /// Execution end time
    pub end_time: Option<SystemTime>,
    /// Total execution duration
    pub duration: Option<Duration>,
    /// Execution status
    pub status: ExecutionStatus,
    /// Input parameters and configuration
    pub parameters: HashMap<String, serde_json::Value>,
    /// Input data references
    pub inputs: Vec<DataReference>,
    /// Output data references
    pub outputs: Vec<DataReference>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Resource usage statistics
    pub resource_usage: ResourceUsage,
    /// Error information (if applicable)
    pub error_info: Option<ErrorInfo>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom attributes
    pub attributes: HashMap<String, serde_json::Value>,
    /// Parent execution ID (for nested executions)
    pub parent_execution_id: Option<String>,
    /// Child execution IDs
    pub child_execution_ids: Vec<String>,
}

/// Execution status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Execution is currently running
    Running,
    /// Execution completed successfully
    Completed,
    /// Execution failed with error
    Failed,
    /// Execution was cancelled
    Cancelled,
    /// Execution timed out
    Timeout,
    /// Execution was skipped
    Skipped,
}

/// Data reference for inputs and outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataReference {
    /// Data identifier
    pub id: String,
    /// Data type
    pub data_type: String,
    /// Data size in bytes
    pub size: Option<usize>,
    /// Data shape (for arrays/tensors)
    pub shape: Option<Vec<usize>>,
    /// Data checksum
    pub checksum: Option<String>,
    /// Storage location
    pub location: Option<String>,
}

/// Performance metrics for executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Throughput (operations per second)
    pub throughput: Option<f64>,
    /// Latency percentiles
    pub latency_p50: Option<Duration>,
    pub latency_p90: Option<Duration>,
    pub latency_p99: Option<Duration>,
    /// Quality metrics
    pub accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1_score: Option<f64>,
    /// Loss function value
    pub loss: Option<f64>,
    /// Custom performance metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage statistics
    pub cpu: CpuUsage,
    /// Memory usage statistics
    pub memory: MemoryUsage,
    /// I/O usage statistics
    pub io: IoUsage,
    /// GPU usage statistics (if applicable)
    pub gpu: Option<GpuUsage>,
    /// Network usage statistics
    pub network: Option<NetworkUsage>,
}

/// CPU usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUsage {
    /// Average CPU utilization percentage
    pub avg_utilization: f64,
    /// Peak CPU utilization percentage
    pub peak_utilization: f64,
    /// Total CPU time consumed
    pub total_cpu_time: Duration,
    /// Number of CPU cores used
    pub cores_used: usize,
    /// CPU cycles per instruction
    pub cycles_per_instruction: Option<f64>,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Average memory usage in bytes
    pub avg_usage: usize,
    /// Total memory allocated in bytes
    pub total_allocated: usize,
    /// Number of memory allocations
    pub allocation_count: usize,
    /// Memory efficiency ratio
    pub efficiency_ratio: f64,
}

/// I/O usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoUsage {
    /// Total bytes read
    pub bytes_read: usize,
    /// Total bytes written
    pub bytes_written: usize,
    /// Number of read operations
    pub read_operations: usize,
    /// Number of write operations
    pub write_operations: usize,
    /// Average I/O latency
    pub avg_latency: Duration,
}

/// GPU usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUsage {
    /// GPU utilization percentage
    pub utilization: f64,
    /// GPU memory usage in bytes
    pub memory_usage: usize,
    /// GPU memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// GPU power consumption in watts
    pub power_consumption: Option<f64>,
}

/// Network usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkUsage {
    /// Total bytes sent
    pub bytes_sent: usize,
    /// Total bytes received
    pub bytes_received: usize,
    /// Number of network connections
    pub connections: usize,
    /// Average network latency
    pub avg_latency: Duration,
}

/// Error information for failed executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error type or category
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Error code
    pub error_code: Option<String>,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
    /// Error occurrence count
    pub occurrence_count: u32,
    /// First occurrence timestamp
    pub first_occurrence: SystemTime,
    /// Last occurrence timestamp
    pub last_occurrence: SystemTime,
}

/// Execution comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionComparison {
    /// Baseline execution ID
    pub baseline_id: String,
    /// Comparison execution ID
    pub comparison_id: String,
    /// Performance delta
    pub performance_delta: PerformanceDelta,
    /// Resource usage delta
    pub resource_delta: ResourceDelta,
    /// Overall improvement score (-100 to 100)
    pub improvement_score: f64,
    /// Comparison timestamp
    pub compared_at: SystemTime,
}

/// Performance improvement delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDelta {
    /// Duration change (negative = improvement)
    pub duration_change: Option<Duration>,
    /// Throughput change percentage
    pub throughput_change_percent: Option<f64>,
    /// Accuracy change
    pub accuracy_change: Option<f64>,
    /// Custom metric changes
    pub custom_metric_changes: HashMap<String, f64>,
}

/// Resource usage delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDelta {
    /// CPU usage change percentage
    pub cpu_change_percent: f64,
    /// Memory usage change percentage
    pub memory_change_percent: f64,
    /// I/O usage change percentage
    pub io_change_percent: f64,
}

/// Execution pattern analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Affected executions
    pub affected_executions: Vec<String>,
    /// Pattern strength (0.0 to 1.0)
    pub strength: f64,
    /// Pattern frequency
    pub frequency: f64,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Types of execution patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Performance degradation over time
    PerformanceDegradation,
    /// Resource usage increase
    ResourceInflation,
    /// Frequent failures
    FailureSpike,
    /// Performance improvement
    PerformanceImprovement,
    /// Unusual resource pattern
    AnomalousResource,
    /// Custom pattern
    Custom(String),
}

/// Execution history configuration
#[derive(Debug, Clone)]
pub struct ExecutionHistoryConfig {
    /// Maximum number of execution records to keep
    pub max_records: usize,
    /// Retention period for execution records
    pub retention_period: Duration,
    /// Enable automatic pattern analysis
    pub enable_pattern_analysis: bool,
    /// Pattern analysis interval
    pub pattern_analysis_interval: Duration,
    /// Enable alerting for anomalies
    pub enable_alerting: bool,
    /// Performance threshold for alerts
    pub performance_alert_threshold: f64,
    /// Resource usage threshold for alerts
    pub resource_alert_threshold: f64,
    /// Error rate threshold for alerts
    pub error_rate_threshold: f64,
}

impl Default for ExecutionHistoryConfig {
    fn default() -> Self {
        Self {
            max_records: 10_000,
            retention_period: Duration::from_secs(86400 * 90), // 90 days
            enable_pattern_analysis: true,
            pattern_analysis_interval: Duration::from_secs(3600), // 1 hour
            enable_alerting: true,
            performance_alert_threshold: 0.2, // 20% degradation
            resource_alert_threshold: 0.3,    // 30% increase
            error_rate_threshold: 0.05,       // 5% error rate
        }
    }
}

/// Execution alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionAlert {
    /// Alert ID
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Affected execution ID
    pub execution_id: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert status
    pub status: AlertStatus,
    /// Recommended actions
    pub actions: Vec<String>,
}

/// Alert types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// Performance degradation alert
    PerformanceDegradation,
    /// High resource usage alert
    HighResourceUsage,
    /// Execution failure alert
    ExecutionFailure,
    /// Anomaly detection alert
    AnomalyDetected,
    /// Custom alert type
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
}

/// Alert status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Alert is active
    Active,
    /// Alert has been acknowledged
    Acknowledged,
    /// Alert has been resolved
    Resolved,
}

/// Main execution history manager
#[derive(Debug)]
pub struct ExecutionHistory {
    /// Configuration
    config: ExecutionHistoryConfig,
    /// Execution records storage
    records: BTreeMap<SystemTime, ExecutionRecord>,
    /// Index by execution ID
    id_index: HashMap<String, SystemTime>,
    /// Index by algorithm name
    algorithm_index: HashMap<String, Vec<SystemTime>>,
    /// Index by status
    status_index: HashMap<ExecutionStatus, Vec<SystemTime>>,
    /// Active executions (still running)
    active_executions: HashMap<String, Instant>,
    /// Error patterns tracking
    error_patterns: HashMap<String, u32>,
    /// Performance baseline
    performance_baselines: HashMap<String, PerformanceMetrics>,
    /// Active alerts
    active_alerts: HashMap<String, ExecutionAlert>,
    /// Performance metrics
    metrics: Arc<MetricRegistry>,
    /// Operation timers
    record_timer: Timer,
    query_timer: Timer,
    analysis_timer: Timer,
    /// Operation counters
    executions_recorded: Counter,
    executions_failed: Counter,
    alerts_generated: Counter,
    /// Performance gauges
    active_executions_gauge: Gauge,
    avg_duration_gauge: Gauge,
    error_rate_gauge: Gauge,
}

impl ExecutionHistory {
    /// Create a new execution history manager
    pub fn new() -> Self {
        Self::with_config(ExecutionHistoryConfig::default())
    }

    /// Create execution history manager with configuration
    pub fn with_config(config: ExecutionHistoryConfig) -> Self {
        let metrics = Arc::new(MetricRegistry::new());

        Self {
            config,
            records: BTreeMap::new(),
            id_index: HashMap::new(),
            algorithm_index: HashMap::new(),
            status_index: HashMap::new(),
            active_executions: HashMap::new(),
            error_patterns: HashMap::new(),
            performance_baselines: HashMap::new(),
            active_alerts: HashMap::new(),
            metrics: metrics.clone(),
            record_timer: metrics.timer("execution_history.record_duration"),
            query_timer: metrics.timer("execution_history.query_duration"),
            analysis_timer: metrics.timer("execution_history.analysis_duration"),
            executions_recorded: metrics.counter("execution_history.executions_recorded"),
            executions_failed: metrics.counter("execution_history.executions_failed"),
            alerts_generated: metrics.counter("execution_history.alerts_generated"),
            active_executions_gauge: metrics.gauge("execution_history.active_executions"),
            avg_duration_gauge: metrics.gauge("execution_history.avg_duration_ms"),
            error_rate_gauge: metrics.gauge("execution_history.error_rate"),
        }
    }

    /// Start tracking a new execution
    pub fn start_execution(
        &mut self,
        algorithm_name: String,
        parameters: HashMap<String, serde_json::Value>,
        inputs: Vec<DataReference>,
        tags: Vec<String>,
        attributes: HashMap<String, serde_json::Value>,
        parent_execution_id: Option<String>,
    ) -> Result<String> {
        let execution_id = Uuid::new_v4().to_string();
        let start_time = SystemTime::now();

        let record = ExecutionRecord {
            execution_id: execution_id.clone(),
            algorithm_name: algorithm_name.clone(),
            start_time,
            end_time: None,
            duration: None,
            status: ExecutionStatus::Running,
            parameters,
            inputs,
            outputs: Vec::new(),
            performance_metrics: PerformanceMetrics {
                throughput: None,
                latency_p50: None,
                latency_p90: None,
                latency_p99: None,
                accuracy: None,
                precision: None,
                recall: None,
                f1_score: None,
                loss: None,
                custom_metrics: HashMap::new(),
            },
            resource_usage: ResourceUsage {
                cpu: CpuUsage {
                    avg_utilization: 0.0,
                    peak_utilization: 0.0,
                    total_cpu_time: Duration::from_secs(0),
                    cores_used: 0,
                    cycles_per_instruction: None,
                },
                memory: MemoryUsage {
                    peak_usage: 0,
                    avg_usage: 0,
                    total_allocated: 0,
                    allocation_count: 0,
                    efficiency_ratio: 0.0,
                },
                io: IoUsage {
                    bytes_read: 0,
                    bytes_written: 0,
                    read_operations: 0,
                    write_operations: 0,
                    avg_latency: Duration::from_secs(0),
                },
                gpu: None,
                network: None,
            },
            error_info: None,
            tags,
            attributes,
            parent_execution_id,
            child_execution_ids: Vec::new(),
        };

        // Store record
        self.records.insert(start_time, record);
        self.id_index.insert(execution_id.clone(), start_time);

        // Update algorithm index
        self.algorithm_index
            .entry(algorithm_name)
            .or_insert_with(Vec::new)
            .push(start_time);

        // Update status index
        self.status_index
            .entry(ExecutionStatus::Running)
            .or_insert_with(Vec::new)
            .push(start_time);

        // Track active execution
        self.active_executions.insert(execution_id.clone(), Instant::now());

        // Update metrics
        self.active_executions_gauge.set(self.active_executions.len() as f64);

        Ok(execution_id)
    }

    /// Complete an execution with results
    pub fn complete_execution(
        &mut self,
        execution_id: &str,
        outputs: Vec<DataReference>,
        performance_metrics: PerformanceMetrics,
        resource_usage: ResourceUsage,
    ) -> Result<()> {
        let _timer = self.record_timer.start_timer();

        let start_time = self.id_index
            .get(execution_id)
            .cloned()
            .ok_or_else(|| CoreError::ValidationError(
                format!("Execution {} not found", execution_id)
            ))?;

        if let Some(record) = self.records.get_mut(&start_time) {
            let end_time = SystemTime::now();
            let duration = end_time.duration_since(record.start_time).unwrap_or_default();

            // Update record
            record.end_time = Some(end_time);
            record.duration = Some(duration);
            record.status = ExecutionStatus::Completed;
            record.outputs = outputs;
            record.performance_metrics = performance_metrics;
            record.resource_usage = resource_usage;

            // Remove from active executions
            self.active_executions.remove(execution_id);

            // Update indexes
            self.remove_from_status_index(&ExecutionStatus::Running, start_time);
            self.add_to_status_index(ExecutionStatus::Completed, start_time);

            // Update performance baseline
            self.update_performance_baseline(&record.algorithm_name, &record.performance_metrics);

            // Update metrics
            self.executions_recorded.inc();
            self.active_executions_gauge.set(self.active_executions.len() as f64);
            self.avg_duration_gauge.set(duration.as_millis() as f64);

            // Check for performance alerts
            if self.config.enable_alerting {
                self.check_performance_alerts(record)?;
            }

            Ok(())
        } else {
            Err(CoreError::ValidationError(
                format!("Execution record {} not found", execution_id)
            ))
        }
    }

    /// Mark an execution as failed
    pub fn fail_execution(
        &mut self,
        execution_id: &str,
        error_info: ErrorInfo,
        resource_usage: Option<ResourceUsage>,
    ) -> Result<()> {
        let _timer = self.record_timer.start_timer();

        let start_time = self.id_index
            .get(execution_id)
            .cloned()
            .ok_or_else(|| CoreError::ValidationError(
                format!("Execution {} not found", execution_id)
            ))?;

        if let Some(record) = self.records.get_mut(&start_time) {
            let end_time = SystemTime::now();
            let duration = end_time.duration_since(record.start_time).unwrap_or_default();

            // Update record
            record.end_time = Some(end_time);
            record.duration = Some(duration);
            record.status = ExecutionStatus::Failed;
            record.error_info = Some(error_info.clone());

            if let Some(usage) = resource_usage {
                record.resource_usage = usage;
            }

            // Remove from active executions
            self.active_executions.remove(execution_id);

            // Update indexes
            self.remove_from_status_index(&ExecutionStatus::Running, start_time);
            self.add_to_status_index(ExecutionStatus::Failed, start_time);

            // Track error patterns
            let error_key = format!("{}:{}", error_info.error_type, record.algorithm_name);
            *self.error_patterns.entry(error_key).or_insert(0) += 1;

            // Update metrics
            self.executions_failed.inc();
            self.active_executions_gauge.set(self.active_executions.len() as f64);

            // Generate failure alert
            if self.config.enable_alerting {
                self.generate_failure_alert(execution_id, &error_info)?;
            }

            Ok(())
        } else {
            Err(CoreError::ValidationError(
                format!("Execution record {} not found", execution_id)
            ))
        }
    }

    /// Get execution record by ID
    pub fn get_execution(&self, execution_id: &str) -> Option<&ExecutionRecord> {
        self.id_index
            .get(execution_id)
            .and_then(|timestamp| self.records.get(timestamp))
    }

    /// Query executions by algorithm name
    pub fn get_executions_by_algorithm(&self, algorithm_name: &str) -> Vec<&ExecutionRecord> {
        self.algorithm_index
            .get(algorithm_name)
            .map(|timestamps| {
                timestamps.iter()
                    .filter_map(|t| self.records.get(t))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Query executions by status
    pub fn get_executions_by_status(&self, status: &ExecutionStatus) -> Vec<&ExecutionRecord> {
        self.status_index
            .get(status)
            .map(|timestamps| {
                timestamps.iter()
                    .filter_map(|t| self.records.get(t))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Query executions by time range
    pub fn get_executions_by_time_range(
        &self,
        start: SystemTime,
        end: SystemTime,
    ) -> Vec<&ExecutionRecord> {
        self.records
            .range(start..=end)
            .map(|(_, record)| record)
            .collect()
    }

    /// Compare two executions
    pub fn compare_executions(
        &self,
        baseline_id: &str,
        comparison_id: &str,
    ) -> Result<ExecutionComparison> {
        let baseline = self.get_execution(baseline_id)
            .ok_or_else(|| CoreError::ValidationError(
                format!("Baseline execution {} not found", baseline_id)
            ))?;

        let comparison = self.get_execution(comparison_id)
            .ok_or_else(|| CoreError::ValidationError(
                format!("Comparison execution {} not found", comparison_id)
            ))?;

        let performance_delta = self.calculate_performance_delta(
            &baseline.performance_metrics,
            &comparison.performance_metrics,
        );

        let resource_delta = self.calculate_resource_delta(
            &baseline.resource_usage,
            &comparison.resource_usage,
        );

        let improvement_score = self.calculate_improvement_score(
            &performance_delta,
            &resource_delta,
        );

        Ok(ExecutionComparison {
            baseline_id: baseline_id.to_string(),
            comparison_id: comparison_id.to_string(),
            performance_delta,
            resource_delta,
            improvement_score,
            compared_at: SystemTime::now(),
        })
    }

    /// Analyze execution patterns
    pub fn analyze_patterns(&self) -> Result<Vec<ExecutionPattern>> {
        let _timer = self.analysis_timer.start_timer();

        let mut patterns = Vec::new();

        // Analyze performance degradation patterns
        patterns.extend(self.analyze_performance_patterns()?);

        // Analyze resource usage patterns
        patterns.extend(self.analyze_resource_patterns()?);

        // Analyze failure patterns
        patterns.extend(self.analyze_failure_patterns()?);

        Ok(patterns)
    }

    /// Get execution statistics
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        stats.insert("total_executions".to_string(), json!(self.records.len()));
        stats.insert("active_executions".to_string(), json!(self.active_executions.len()));

        // Status distribution
        let mut status_counts = HashMap::new();
        for (status, timestamps) in &self.status_index {
            status_counts.insert(format!("{:?}", status), timestamps.len());
        }
        stats.insert("status_distribution".to_string(), json!(status_counts));

        // Algorithm distribution
        let mut algorithm_counts = HashMap::new();
        for (algorithm, timestamps) in &self.algorithm_index {
            algorithm_counts.insert(algorithm.clone(), timestamps.len());
        }
        stats.insert("algorithm_distribution".to_string(), json!(algorithm_counts));

        // Error patterns
        stats.insert("error_patterns".to_string(), json!(self.error_patterns));

        // Performance metrics
        let avg_duration = self.calculate_average_duration();
        stats.insert("average_duration_ms".to_string(),
                    json!(avg_duration.as_millis()));

        let error_rate = self.calculate_error_rate();
        stats.insert("error_rate".to_string(), json!(error_rate));

        stats
    }

    /// Export execution history
    pub fn export_history(&self) -> Result<String> {
        let export_data = serde_json::json!({
            "version": "1.0",
            "timestamp": SystemTime::now(),
            "executions": self.records.values().collect::<Vec<_>>(),
            "error_patterns": self.error_patterns,
            "performance_baselines": self.performance_baselines,
            "statistics": self.get_statistics()
        });

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| CoreError::SerializationError(format!("Export failed: {}", e)))
    }

    // Private helper methods

    fn remove_from_status_index(&mut self, status: &ExecutionStatus, timestamp: SystemTime) {
        if let Some(timestamps) = self.status_index.get_mut(status) {
            timestamps.retain(|&t| t != timestamp);
            if timestamps.is_empty() {
                self.status_index.remove(status);
            }
        }
    }

    fn add_to_status_index(&mut self, status: ExecutionStatus, timestamp: SystemTime) {
        self.status_index
            .entry(status)
            .or_insert_with(Vec::new)
            .push(timestamp);
    }

    fn update_performance_baseline(
        &mut self,
        algorithm_name: &str,
        metrics: &PerformanceMetrics,
    ) {
        // Simple baseline update - in production would use more sophisticated averaging
        self.performance_baselines.insert(algorithm_name.to_string(), metrics.clone());
    }

    fn check_performance_alerts(&mut self, record: &ExecutionRecord) -> Result<()> {
        if let Some(baseline) = self.performance_baselines.get(&record.algorithm_name) {
            // Check for performance degradation
            if let (Some(current_duration), Some(baseline_duration)) =
                (record.duration, baseline.latency_p50) {

                let degradation = current_duration.as_secs_f64() / baseline_duration.as_secs_f64() - 1.0;

                if degradation > self.config.performance_alert_threshold {
                    self.generate_performance_alert(&record.execution_id, degradation)?;
                }
            }
        }

        Ok(())
    }

    fn generate_failure_alert(&mut self, execution_id: &str, error_info: &ErrorInfo) -> Result<()> {
        let alert = ExecutionAlert {
            id: Uuid::new_v4().to_string(),
            alert_type: AlertType::ExecutionFailure,
            severity: AlertSeverity::Warning,
            message: format!("Execution failed: {}", error_info.message),
            execution_id: execution_id.to_string(),
            timestamp: SystemTime::now(),
            status: AlertStatus::Active,
            actions: error_info.recovery_suggestions.clone(),
        };

        self.active_alerts.insert(alert.id.clone(), alert);
        self.alerts_generated.inc();

        Ok(())
    }

    fn generate_performance_alert(&mut self, execution_id: &str, degradation: f64) -> Result<()> {
        let alert = ExecutionAlert {
            id: Uuid::new_v4().to_string(),
            alert_type: AlertType::PerformanceDegradation,
            severity: if degradation > 0.5 { AlertSeverity::Critical } else { AlertSeverity::Warning },
            message: format!("Performance degradation detected: {:.1}%", degradation * 100.0),
            execution_id: execution_id.to_string(),
            timestamp: SystemTime::now(),
            status: AlertStatus::Active,
            actions: vec![
                "Review recent changes".to_string(),
                "Check resource usage".to_string(),
                "Consider performance optimization".to_string(),
            ],
        };

        self.active_alerts.insert(alert.id.clone(), alert);
        self.alerts_generated.inc();

        Ok(())
    }

    fn calculate_performance_delta(
        &self,
        baseline: &PerformanceMetrics,
        comparison: &PerformanceMetrics,
    ) -> PerformanceDelta {
        PerformanceDelta {
            duration_change: None, // Would calculate from actual duration data
            throughput_change_percent: match (baseline.throughput, comparison.throughput) {
                (Some(b), Some(c)) => Some((c - b) / b * 100.0),
                _ => None,
            },
            accuracy_change: match (baseline.accuracy, comparison.accuracy) {
                (Some(b), Some(c)) => Some(c - b),
                _ => None,
            },
            custom_metric_changes: HashMap::new(), // Would compare custom metrics
        }
    }

    fn calculate_resource_delta(
        &self,
        baseline: &ResourceUsage,
        comparison: &ResourceUsage,
    ) -> ResourceDelta {
        ResourceDelta {
            cpu_change_percent: (comparison.cpu.avg_utilization - baseline.cpu.avg_utilization) / baseline.cpu.avg_utilization * 100.0,
            memory_change_percent: (comparison.memory.avg_usage as f64 - baseline.memory.avg_usage as f64) / baseline.memory.avg_usage as f64 * 100.0,
            io_change_percent: (comparison.io.bytes_read as f64 - baseline.io.bytes_read as f64) / baseline.io.bytes_read.max(1) as f64 * 100.0,
        }
    }

    fn calculate_improvement_score(
        &self,
        performance_delta: &PerformanceDelta,
        resource_delta: &ResourceDelta,
    ) -> f64 {
        let mut score = 0.0;

        // Performance improvements add to score
        if let Some(throughput_change) = performance_delta.throughput_change_percent {
            score += throughput_change * 0.3; // Weight: 30%
        }

        if let Some(accuracy_change) = performance_delta.accuracy_change {
            score += accuracy_change * 100.0 * 0.4; // Weight: 40%
        }

        // Resource efficiency improvements add to score
        score -= resource_delta.cpu_change_percent * 0.1; // Weight: 10%
        score -= resource_delta.memory_change_percent * 0.1; // Weight: 10%
        score -= resource_delta.io_change_percent * 0.1; // Weight: 10%

        // Clamp score to [-100, 100]
        score.max(-100.0).min(100.0)
    }

    fn analyze_performance_patterns(&self) -> Result<Vec<ExecutionPattern>> {
        let mut patterns = Vec::new();

        // Look for performance degradation trends
        for (algorithm, _) in &self.algorithm_index {
            let executions = self.get_executions_by_algorithm(algorithm);

            if executions.len() >= 5 {
                let recent_avg = self.calculate_recent_average_duration(&executions, 5);
                let historical_avg = self.calculate_historical_average_duration(&executions, 10);

                if let (Some(recent), Some(historical)) = (recent_avg, historical_avg) {
                    let degradation = recent.as_secs_f64() / historical.as_secs_f64() - 1.0;

                    if degradation > 0.15 { // 15% degradation
                        patterns.push(ExecutionPattern {
                            pattern_type: PatternType::PerformanceDegradation,
                            description: format!("Performance degradation detected for {}", algorithm),
                            affected_executions: executions.iter().take(5).map(|e| e.execution_id.clone()).collect(),
                            strength: degradation.min(1.0),
                            frequency: 1.0,
                            recommendations: vec![
                                "Investigate recent changes".to_string(),
                                "Check for resource constraints".to_string(),
                                "Consider performance optimization".to_string(),
                            ],
                        });
                    }
                }
            }
        }

        Ok(patterns)
    }

    fn analyze_resource_patterns(&self) -> Result<Vec<ExecutionPattern>> {
        let mut patterns = Vec::new();

        // Analyze memory usage trends
        for (algorithm, _) in &self.algorithm_index {
            let executions = self.get_executions_by_algorithm(algorithm);

            if executions.len() >= 5 {
                let recent_memory = self.calculate_recent_average_memory(&executions, 5);
                let historical_memory = self.calculate_historical_average_memory(&executions, 10);

                if let (Some(recent), Some(historical)) = (recent_memory, historical_memory) {
                    let increase = recent as f64 / historical as f64 - 1.0;

                    if increase > 0.2 { // 20% increase
                        patterns.push(ExecutionPattern {
                            pattern_type: PatternType::ResourceInflation,
                            description: format!("Memory usage inflation detected for {}", algorithm),
                            affected_executions: executions.iter().take(5).map(|e| e.execution_id.clone()).collect(),
                            strength: increase.min(1.0),
                            frequency: 1.0,
                            recommendations: vec![
                                "Check for memory leaks".to_string(),
                                "Review data size changes".to_string(),
                                "Consider memory optimization".to_string(),
                            ],
                        });
                    }
                }
            }
        }

        Ok(patterns)
    }

    fn analyze_failure_patterns(&self) -> Result<Vec<ExecutionPattern>> {
        let mut patterns = Vec::new();

        // Calculate recent failure rates
        for (algorithm, _) in &self.algorithm_index {
            let executions = self.get_executions_by_algorithm(algorithm);

            if executions.len() >= 10 {
                let recent_failures = executions.iter()
                    .take(10)
                    .filter(|e| e.status == ExecutionStatus::Failed)
                    .count();

                let failure_rate = recent_failures as f64 / 10.0;

                if failure_rate > 0.1 { // 10% failure rate
                    patterns.push(ExecutionPattern {
                        pattern_type: PatternType::FailureSpike,
                        description: format!("High failure rate detected for {}: {:.1}%", algorithm, failure_rate * 100.0),
                        affected_executions: executions.iter()
                            .filter(|e| e.status == ExecutionStatus::Failed)
                            .take(5)
                            .map(|e| e.execution_id.clone())
                            .collect(),
                        strength: failure_rate,
                        frequency: failure_rate,
                        recommendations: vec![
                            "Investigate common failure causes".to_string(),
                            "Review input validation".to_string(),
                            "Check infrastructure health".to_string(),
                        ],
                    });
                }
            }
        }

        Ok(patterns)
    }

    fn calculate_recent_average_duration(&self, executions: &[&ExecutionRecord], count: usize) -> Option<Duration> {
        let recent_durations: Vec<_> = executions.iter()
            .take(count)
            .filter_map(|e| e.duration)
            .collect();

        if recent_durations.is_empty() {
            return None;
        }

        let total_ms: u64 = recent_durations.iter().map(|d| d.as_millis() as u64).sum();
        Some(Duration::from_millis(total_ms / recent_durations.len() as u64))
    }

    fn calculate_historical_average_duration(&self, executions: &[&ExecutionRecord], start_from: usize) -> Option<Duration> {
        if executions.len() <= start_from {
            return None;
        }

        let historical_durations: Vec<_> = executions.iter()
            .skip(start_from)
            .filter_map(|e| e.duration)
            .collect();

        if historical_durations.is_empty() {
            return None;
        }

        let total_ms: u64 = historical_durations.iter().map(|d| d.as_millis() as u64).sum();
        Some(Duration::from_millis(total_ms / historical_durations.len() as u64))
    }

    fn calculate_recent_average_memory(&self, executions: &[&ExecutionRecord], count: usize) -> Option<usize> {
        let recent_memory: Vec<_> = executions.iter()
            .take(count)
            .map(|e| e.resource_usage.memory.avg_usage)
            .collect();

        if recent_memory.is_empty() {
            None
        } else {
            Some(recent_memory.iter().sum::<usize>() / recent_memory.len())
        }
    }

    fn calculate_historical_average_memory(&self, executions: &[&ExecutionRecord], start_from: usize) -> Option<usize> {
        if executions.len() <= start_from {
            return None;
        }

        let historical_memory: Vec<_> = executions.iter()
            .skip(start_from)
            .map(|e| e.resource_usage.memory.avg_usage)
            .collect();

        if historical_memory.is_empty() {
            None
        } else {
            Some(historical_memory.iter().sum::<usize>() / historical_memory.len())
        }
    }

    fn calculate_average_duration(&self) -> Duration {
        let durations: Vec<_> = self.records.values()
            .filter_map(|r| r.duration)
            .collect();

        if durations.is_empty() {
            return Duration::from_secs(0);
        }

        let total_ms: u64 = durations.iter().map(|d| d.as_millis() as u64).sum();
        Duration::from_millis(total_ms / durations.len() as u64)
    }

    fn calculate_error_rate(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }

        let failed_count = self.records.values()
            .filter(|r| r.status == ExecutionStatus::Failed)
            .count();

        failed_count as f64 / self.records.len() as f64
    }
}

impl Default for ExecutionHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_execution_lifecycle() {
        let mut history = ExecutionHistory::new();

        // Start execution
        let execution_id = history.start_execution(
            "test_algorithm".to_string(),
            HashMap::new(),
            vec![],
            vec!["test".to_string()],
            HashMap::new(),
            None,
        ).unwrap();

        // Verify execution is tracked
        let record = history.get_execution(&execution_id).unwrap();
        assert_eq!(record.status, ExecutionStatus::Running);
        assert_eq!(record.algorithm_name, "test_algorithm");

        // Complete execution
        let performance_metrics = PerformanceMetrics {
            throughput: Some(100.0),
            latency_p50: Some(Duration::from_millis(50)),
            latency_p90: Some(Duration::from_millis(90)),
            latency_p99: Some(Duration::from_millis(99)),
            accuracy: Some(0.95),
            precision: Some(0.92),
            recall: Some(0.88),
            f1_score: Some(0.90),
            loss: Some(0.05),
            custom_metrics: HashMap::new(),
        };

        let resource_usage = ResourceUsage {
            cpu: CpuUsage {
                avg_utilization: 50.0,
                peak_utilization: 80.0,
                total_cpu_time: Duration::from_secs(10),
                cores_used: 4,
                cycles_per_instruction: Some(2.5),
            },
            memory: MemoryUsage {
                peak_usage: 1024 * 1024 * 100, // 100MB
                avg_usage: 1024 * 1024 * 80,   // 80MB
                total_allocated: 1024 * 1024 * 200,
                allocation_count: 1000,
                efficiency_ratio: 0.8,
            },
            io: IoUsage {
                bytes_read: 1024 * 1024,
                bytes_written: 512 * 1024,
                read_operations: 100,
                write_operations: 50,
                avg_latency: Duration::from_millis(5),
            },
            gpu: None,
            network: None,
        };

        history.complete_execution(
            &execution_id,
            vec![],
            performance_metrics,
            resource_usage,
        ).unwrap();

        // Verify completion
        let completed_record = history.get_execution(&execution_id).unwrap();
        assert_eq!(completed_record.status, ExecutionStatus::Completed);
        assert!(completed_record.duration.is_some());
        assert_eq!(completed_record.performance_metrics.accuracy, Some(0.95));
    }

    #[test]
    fn test_execution_failure() {
        let mut history = ExecutionHistory::new();

        let execution_id = history.start_execution(
            "failing_algorithm".to_string(),
            HashMap::new(),
            vec![],
            vec![],
            HashMap::new(),
            None,
        ).unwrap();

        let error_info = ErrorInfo {
            error_type: "ValidationError".to_string(),
            message: "Invalid input data".to_string(),
            stack_trace: None,
            error_code: Some("E001".to_string()),
            recovery_suggestions: vec!["Check input format".to_string()],
            occurrence_count: 1,
            first_occurrence: SystemTime::now(),
            last_occurrence: SystemTime::now(),
        };

        history.fail_execution(&execution_id, error_info, None).unwrap();

        let failed_record = history.get_execution(&execution_id).unwrap();
        assert_eq!(failed_record.status, ExecutionStatus::Failed);
        assert!(failed_record.error_info.is_some());
    }

    #[test]
    fn test_execution_queries() {
        let mut history = ExecutionHistory::new();

        // Create multiple executions
        let id1 = history.start_execution(
            "algorithm_a".to_string(),
            HashMap::new(),
            vec![],
            vec![],
            HashMap::new(),
            None,
        ).unwrap();

        let id2 = history.start_execution(
            "algorithm_b".to_string(),
            HashMap::new(),
            vec![],
            vec![],
            HashMap::new(),
            None,
        ).unwrap();

        // Complete one, fail another
        history.complete_execution(
            &id1,
            vec![],
            PerformanceMetrics {
                throughput: None,
                latency_p50: None,
                latency_p90: None,
                latency_p99: None,
                accuracy: None,
                precision: None,
                recall: None,
                f1_score: None,
                loss: None,
                custom_metrics: HashMap::new(),
            },
            ResourceUsage {
                cpu: CpuUsage {
                    avg_utilization: 0.0,
                    peak_utilization: 0.0,
                    total_cpu_time: Duration::from_secs(0),
                    cores_used: 0,
                    cycles_per_instruction: None,
                },
                memory: MemoryUsage {
                    peak_usage: 0,
                    avg_usage: 0,
                    total_allocated: 0,
                    allocation_count: 0,
                    efficiency_ratio: 0.0,
                },
                io: IoUsage {
                    bytes_read: 0,
                    bytes_written: 0,
                    read_operations: 0,
                    write_operations: 0,
                    avg_latency: Duration::from_secs(0),
                },
                gpu: None,
                network: None,
            },
        ).unwrap();

        history.fail_execution(
            &id2,
            ErrorInfo {
                error_type: "TestError".to_string(),
                message: "Test failure".to_string(),
                stack_trace: None,
                error_code: None,
                recovery_suggestions: vec![],
                occurrence_count: 1,
                first_occurrence: SystemTime::now(),
                last_occurrence: SystemTime::now(),
            },
            None,
        ).unwrap();

        // Test queries
        let algorithm_a_executions = history.get_executions_by_algorithm("algorithm_a");
        assert_eq!(algorithm_a_executions.len(), 1);

        let completed_executions = history.get_executions_by_status(&ExecutionStatus::Completed);
        assert_eq!(completed_executions.len(), 1);

        let failed_executions = history.get_executions_by_status(&ExecutionStatus::Failed);
        assert_eq!(failed_executions.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let mut history = ExecutionHistory::new();

        let id = history.start_execution(
            "test_algorithm".to_string(),
            HashMap::new(),
            vec![],
            vec![],
            HashMap::new(),
            None,
        ).unwrap();

        history.complete_execution(
            &id,
            vec![],
            PerformanceMetrics {
                throughput: None,
                latency_p50: None,
                latency_p90: None,
                latency_p99: None,
                accuracy: None,
                precision: None,
                recall: None,
                f1_score: None,
                loss: None,
                custom_metrics: HashMap::new(),
            },
            ResourceUsage {
                cpu: CpuUsage {
                    avg_utilization: 0.0,
                    peak_utilization: 0.0,
                    total_cpu_time: Duration::from_secs(0),
                    cores_used: 0,
                    cycles_per_instruction: None,
                },
                memory: MemoryUsage {
                    peak_usage: 0,
                    avg_usage: 0,
                    total_allocated: 0,
                    allocation_count: 0,
                    efficiency_ratio: 0.0,
                },
                io: IoUsage {
                    bytes_read: 0,
                    bytes_written: 0,
                    read_operations: 0,
                    write_operations: 0,
                    avg_latency: Duration::from_secs(0),
                },
                gpu: None,
                network: None,
            },
        ).unwrap();

        let stats = history.get_statistics();
        assert_eq!(stats["total_executions"], json!(1));
        assert_eq!(stats["active_executions"], json!(0));
    }
}