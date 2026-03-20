//! Core Monitoring Traits and Interfaces
//!
//! This module defines the fundamental traits, interfaces, and core types that form
//! the foundation of the execution monitoring framework. It provides the contract
//! that all monitoring implementations must follow.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;

use crate::execution_types::*;
use crate::task_scheduling::{TaskHandle, TaskState};
use crate::resource_management::{ResourceAllocation, ResourceUtilization};
use crate::session_management::{MonitoringSession, MonitoringSessionState};
use crate::metrics_collection::{PerformanceMetric, MetricsStorage};
use crate::event_tracking::{TaskExecutionEvent, EventBuffer};
use crate::alerting_system::AlertManager;
use crate::health_monitoring::HealthChecker;
use crate::configuration_management::*;
use crate::reporting_system::{MonitoringReport, ReportConfig};
use crate::data_retention::{ExportConfig, DataExporter};

/// Core execution monitor trait for pluggable monitoring implementations
///
/// This trait provides a flexible interface for different monitoring strategies
/// that can collect metrics, track performance, and provide comprehensive observability
/// for execution engine operations.
///
/// # Design Principles
///
/// - **Pluggability**: Support for different monitoring backends and strategies
/// - **Performance**: Minimal overhead monitoring with configurable sampling
/// - **Scalability**: Handle multiple concurrent monitoring sessions
/// - **Reliability**: Fault-tolerant monitoring with graceful degradation
/// - **Extensibility**: Support for custom metrics, events, and analysis
///
/// # Implementation Guidelines
///
/// Implementing types should:
/// - Minimize performance overhead on monitored systems
/// - Provide thread-safe operations for concurrent access
/// - Handle resource constraints gracefully
/// - Support configuration updates without restart
/// - Implement proper cleanup and resource management
pub trait ExecutionMonitor: Send + Sync {
    /// Start monitoring for a specific execution session
    ///
    /// Creates a new monitoring session with the specified configuration and begins
    /// collecting metrics, events, and health data for the session.
    ///
    /// # Arguments
    /// * `session_id` - Unique identifier for the execution session
    /// * `config` - Monitoring configuration specifying behavior and capabilities
    ///
    /// # Returns
    /// A monitoring session handle for tracking and control operations
    ///
    /// # Errors
    /// - `InvalidSession` if session_id is invalid or already exists
    /// - `ConfigurationError` if the monitoring configuration is invalid
    /// - `ResourceError` if insufficient resources to start monitoring
    fn start_monitoring(&mut self, session_id: String, config: MonitoringConfig) -> SklResult<MonitoringSession>;

    /// Stop monitoring for a specific session
    ///
    /// Gracefully stops monitoring for the specified session, finalizes data collection,
    /// and generates a comprehensive final monitoring report.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier to stop monitoring for
    ///
    /// # Returns
    /// Final monitoring report containing complete session analysis
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `SessionStateError` if the session is not in a stoppable state
    fn stop_monitoring(&mut self, session_id: String) -> SklResult<MonitoringReport>;

    /// Record a task execution event
    ///
    /// Records an execution event for tracking task lifecycle, performance,
    /// and behavior patterns within the monitoring session.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `event` - Task execution event containing detailed information
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `BufferOverflow` if the event buffer is full
    fn record_task_event(&mut self, session_id: String, event: TaskExecutionEvent) -> SklResult<()>;

    /// Record a resource utilization snapshot
    ///
    /// Records current resource utilization metrics including CPU, memory,
    /// I/O, and custom resources for performance analysis.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `utilization` - Resource utilization snapshot
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `InvalidData` if utilization data is malformed
    fn record_resource_utilization(&mut self, session_id: String, utilization: ResourceUtilization) -> SklResult<()>;

    /// Record a performance metric
    ///
    /// Records a performance metric value for analysis, alerting, and
    /// trend tracking within the monitoring session.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `metric` - Performance metric with value, timestamp, and metadata
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `InvalidMetric` if the metric data is invalid
    fn record_performance_metric(&mut self, session_id: String, metric: PerformanceMetric) -> SklResult<()>;

    /// Get current monitoring status for a session
    ///
    /// Retrieves real-time monitoring status including active metrics,
    /// alerts, health status, and performance summary.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    ///
    /// # Returns
    /// Current monitoring status with real-time information
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    fn get_monitoring_status(&self, session_id: &str) -> SklResult<MonitoringStatus>;

    /// Get historical monitoring data
    ///
    /// Retrieves historical monitoring data within the specified time range
    /// for analysis, reporting, and trend identification.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `time_range` - Time range for historical data retrieval
    ///
    /// # Returns
    /// Historical monitoring data within the specified time range
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `InvalidTimeRange` if the time range is invalid
    fn get_historical_data(&self, session_id: &str, time_range: TimeRange) -> SklResult<HistoricalMonitoringData>;

    /// Generate comprehensive monitoring report
    ///
    /// Generates a detailed monitoring report with analysis, insights,
    /// recommendations, and visualizations based on collected data.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `report_config` - Report generation configuration
    ///
    /// # Returns
    /// Comprehensive monitoring report
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `ReportGenerationError` if report generation fails
    fn generate_report(&self, session_id: &str, report_config: ReportConfig) -> SklResult<MonitoringReport>;

    /// Configure alert thresholds and rules
    ///
    /// Updates alerting configuration for the monitoring session including
    /// thresholds, rules, channels, and suppression settings.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `alert_config` - Alert configuration with rules and settings
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `InvalidConfig` if the alert configuration is invalid
    fn configure_alerts(&mut self, session_id: String, alert_config: AlertConfig) -> SklResult<()>;

    /// Update monitoring configuration
    ///
    /// Updates the monitoring configuration for an active session without
    /// interrupting data collection.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `config` - Updated monitoring configuration
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `ConfigurationError` if the configuration is invalid
    fn update_config(&mut self, session_id: String, config: MonitoringConfig) -> SklResult<()>;

    /// Export monitoring data
    ///
    /// Exports monitoring data in the specified format for external analysis,
    /// archival, or integration with other systems.
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `export_config` - Export configuration specifying format and options
    ///
    /// # Returns
    /// Exported monitoring data in the specified format
    ///
    /// # Errors
    /// - `SessionNotFound` if the session doesn't exist
    /// - `ExportError` if data export fails
    fn export_data(&self, session_id: &str, export_config: ExportConfig) -> SklResult<String>;

    // Optional methods with default implementations

    /// Pause monitoring for a session
    ///
    /// Temporarily pauses data collection while maintaining session state.
    /// Can be resumed later without data loss.
    fn pause_monitoring(&mut self, session_id: String) -> SklResult<()> {
        // Default implementation - update session status
        Ok(())
    }

    /// Resume monitoring for a paused session
    ///
    /// Resumes data collection for a previously paused session.
    fn resume_monitoring(&mut self, session_id: String) -> SklResult<()> {
        // Default implementation - update session status
        Ok(())
    }

    /// Get framework capabilities
    ///
    /// Returns the monitoring capabilities supported by this implementation.
    fn get_capabilities(&self) -> MonitoringCapabilities {
        MonitoringCapabilities::default()
    }

    /// Perform health check on the monitoring system
    ///
    /// Checks the health of the monitoring system itself.
    fn health_check(&self) -> SklResult<SystemHealth> {
        Ok(SystemHealth {
            status: HealthStatus::Healthy,
            components: HashMap::new(),
            score: 1.0,
            issues: Vec::new(),
        })
    }
}

/// Monitoring status information
///
/// Provides real-time monitoring status including metrics, alerts,
/// health information, and performance summaries.
#[derive(Debug, Clone)]
pub struct MonitoringStatus {
    /// Session identifier
    pub session_id: String,

    /// Session status
    pub session_status: MonitoringSessionStatus,

    /// Last update timestamp
    pub last_update: SystemTime,

    /// Real-time metrics snapshot
    pub real_time_metrics: Vec<PerformanceMetric>,

    /// Active alerts
    pub active_alerts: Vec<ActiveAlert>,

    /// System health status
    pub system_health: SystemHealth,

    /// Resource utilization
    pub resource_utilization: ResourceUtilization,

    /// Performance summary
    pub performance_summary: PerformanceSummary,
}

/// Historical monitoring data
///
/// Contains historical monitoring data for a specified time range
/// including metrics, events, alerts, and derived insights.
#[derive(Debug, Clone)]
pub struct HistoricalMonitoringData {
    /// Session identifier
    pub session_id: String,

    /// Time range of the data
    pub time_range: TimeRange,

    /// Historical metrics
    pub metrics: Vec<PerformanceMetric>,

    /// Historical events
    pub events: Vec<TaskExecutionEvent>,

    /// Historical alerts
    pub alerts: Vec<AlertRecord>,

    /// Resource utilization history
    pub resource_history: Vec<ResourceUtilization>,

    /// Data summary statistics
    pub summary: DataSummary,
}

/// Time range specification
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Start time (inclusive)
    pub start: SystemTime,

    /// End time (inclusive)
    pub end: SystemTime,
}

impl TimeRange {
    /// Create new time range
    pub fn new(start: SystemTime, end: SystemTime) -> Self {
        Self { start, end }
    }

    /// Create time range for last N seconds
    pub fn last_seconds(seconds: u64) -> Self {
        let end = SystemTime::now();
        let start = end - Duration::from_secs(seconds);
        Self { start, end }
    }

    /// Create time range for last N minutes
    pub fn last_minutes(minutes: u64) -> Self {
        Self::last_seconds(minutes * 60)
    }

    /// Create time range for last N hours
    pub fn last_hours(hours: u64) -> Self {
        Self::last_seconds(hours * 3600)
    }

    /// Check if timestamp is within range
    pub fn contains(&self, timestamp: SystemTime) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }

    /// Get duration of the time range
    pub fn duration(&self) -> Duration {
        self.end.duration_since(self.start).unwrap_or(Duration::from_secs(0))
    }
}

/// Alert record for historical data
#[derive(Debug, Clone)]
pub struct AlertRecord {
    /// Alert identifier
    pub alert_id: String,

    /// Rule that triggered the alert
    pub rule_name: String,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Alert trigger time
    pub triggered_at: SystemTime,

    /// Alert resolution time (if resolved)
    pub resolved_at: Option<SystemTime>,

    /// Alert message
    pub message: String,

    /// Associated metric values
    pub metric_values: HashMap<String, f64>,
}

/// Data summary statistics
#[derive(Debug, Clone)]
pub struct DataSummary {
    /// Total metrics collected
    pub total_metrics: u64,

    /// Total events recorded
    pub total_events: u64,

    /// Total alerts triggered
    pub total_alerts: u64,

    /// Data collection rate (items per second)
    pub collection_rate: f64,

    /// Data quality score (0.0 to 1.0)
    pub quality_score: f64,

    /// Coverage percentage (0.0 to 1.0)
    pub coverage: f64,
}

/// Default execution monitor implementation
///
/// A comprehensive monitoring implementation that provides all standard
/// monitoring features including metrics collection, event tracking,
/// alerting, health monitoring, and performance analysis.
#[derive(Debug)]
pub struct DefaultExecutionMonitor {
    /// Active monitoring sessions
    pub sessions: HashMap<String, MonitoringSessionState>,

    /// Global monitoring configuration
    pub global_config: MonitoringConfig,

    /// Metrics storage backend
    pub metrics_storage: Box<dyn MetricsStorage>,

    /// Event buffer for processing
    pub event_buffer: VecDeque<(String, TaskExecutionEvent)>,

    /// Alert manager
    pub alert_manager: AlertManager,

    /// Health checker
    pub health_checker: HealthChecker,
}

/// Monitoring session status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum MonitoringSessionStatus {
    /// Session is starting up
    Starting,
    /// Session is actively monitoring
    Active,
    /// Session is paused
    Paused,
    /// Session is stopping
    Stopping,
    /// Session has stopped
    Stopped,
    /// Session has failed
    Failed { reason: String },
}

/// Active alert information
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// Alert identifier
    pub alert_id: String,

    /// Alert rule name
    pub rule_name: String,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Alert message
    pub message: String,

    /// Alert start time
    pub start_time: SystemTime,

    /// Alert acknowledgment status
    pub acknowledged: bool,
}

/// System health status
#[derive(Debug, Clone)]
pub struct SystemHealth {
    /// Overall health status
    pub status: HealthStatus,

    /// Component health mapping
    pub components: HashMap<String, ComponentHealth>,

    /// Overall health score (0.0 to 1.0)
    pub score: f64,

    /// Current health issues
    pub issues: Vec<HealthIssue>,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System has warnings
    Warning,
    /// System is in critical state
    Critical,
    /// Health status unknown
    Unknown,
}

/// Component health information
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// Component name
    pub component: String,

    /// Component health status
    pub status: HealthStatus,

    /// Component health score (0.0 to 1.0)
    pub score: f64,

    /// Last health check time
    pub last_check: SystemTime,

    /// Component issues
    pub issues: Vec<String>,
}

/// Health issue information
#[derive(Debug, Clone)]
pub struct HealthIssue {
    /// Issue identifier
    pub issue_id: String,

    /// Issue type
    pub issue_type: String,

    /// Issue description
    pub description: String,

    /// Issue severity
    pub severity: SeverityLevel,

    /// First occurrence time
    pub first_occurrence: SystemTime,

    /// Issue occurrence count
    pub count: usize,
}

/// Performance summary information
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Average execution time
    pub avg_execution_time: Duration,

    /// Total tasks processed
    pub total_tasks: u64,

    /// Task success rate (0.0 to 1.0)
    pub success_rate: f64,

    /// Throughput (tasks per second)
    pub throughput: f64,

    /// Resource efficiency (0.0 to 1.0)
    pub resource_efficiency: f64,

    /// Performance trends
    pub trends: PerformanceTrends,
}

/// Performance trends information
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Execution time trend
    pub execution_time_trend: TrendDirection,

    /// Throughput trend
    pub throughput_trend: TrendDirection,

    /// Error rate trend
    pub error_rate_trend: TrendDirection,

    /// Resource utilization trend
    pub resource_utilization_trend: TrendDirection,
}

/// Trend direction enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Performance is improving
    Improving,
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading,
    /// Trend is unknown
    Unknown,
}

/// Severity level enumeration
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SeverityLevel {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Monitoring capabilities information
#[derive(Debug, Clone)]
pub struct MonitoringCapabilities {
    /// Supports real-time metrics
    pub real_time_metrics: bool,

    /// Supports historical data
    pub historical_data: bool,

    /// Supports alerting
    pub alerting: bool,

    /// Supports health monitoring
    pub health_monitoring: bool,

    /// Supports performance analysis
    pub performance_analysis: bool,

    /// Supports anomaly detection
    pub anomaly_detection: bool,

    /// Supports custom metrics
    pub custom_metrics: bool,

    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,

    /// Supported export formats
    pub export_formats: Vec<String>,
}

impl Default for MonitoringCapabilities {
    fn default() -> Self {
        Self {
            real_time_metrics: true,
            historical_data: true,
            alerting: true,
            health_monitoring: true,
            performance_analysis: true,
            anomaly_detection: true,
            custom_metrics: true,
            max_concurrent_sessions: 100,
            export_formats: vec![
                "JSON".to_string(),
                "CSV".to_string(),
                "Parquet".to_string(),
                "ProtoBuf".to_string(),
            ],
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range() {
        let start = SystemTime::now();
        let end = start + Duration::from_secs(60);
        let range = TimeRange::new(start, end);

        assert_eq!(range.duration(), Duration::from_secs(60));
        assert!(range.contains(start + Duration::from_secs(30)));
        assert!(!range.contains(start + Duration::from_secs(90)));
    }

    #[test]
    fn test_time_range_convenience() {
        let range = TimeRange::last_minutes(5);
        assert!(range.duration().as_secs() >= 300);

        let range = TimeRange::last_hours(1);
        assert!(range.duration().as_secs() >= 3600);
    }

    #[test]
    fn test_monitoring_capabilities() {
        let capabilities = MonitoringCapabilities::default();
        assert!(capabilities.real_time_metrics);
        assert!(capabilities.alerting);
        assert!(capabilities.max_concurrent_sessions > 0);
        assert!(!capabilities.export_formats.is_empty());
    }

    #[test]
    fn test_severity_levels() {
        assert!(SeverityLevel::Critical > SeverityLevel::High);
        assert!(SeverityLevel::High > SeverityLevel::Medium);
        assert!(SeverityLevel::Medium > SeverityLevel::Low);
    }

    #[test]
    fn test_health_status() {
        let status = HealthStatus::Healthy;
        assert_eq!(status, HealthStatus::Healthy);

        let health = SystemHealth {
            status: HealthStatus::Warning,
            components: HashMap::new(),
            score: 0.85,
            issues: Vec::new(),
        };
        assert_eq!(health.status, HealthStatus::Warning);
        assert_eq!(health.score, 0.85);
    }

    #[test]
    fn test_monitoring_session_status() {
        let status = MonitoringSessionStatus::Active;
        assert_eq!(status, MonitoringSessionStatus::Active);

        let failed_status = MonitoringSessionStatus::Failed {
            reason: "Test failure".to_string(),
        };
        assert!(matches!(failed_status, MonitoringSessionStatus::Failed { .. }));
    }
}