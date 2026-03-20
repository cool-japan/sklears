//! Monitoring and Metrics System
//!
//! This module handles performance monitoring, metrics collection,
//! health monitoring, alerting, and notification systems for report generation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Report generation metrics collector
///
/// Aggregates performance metrics, resource utilization,
/// and error statistics for system monitoring and optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportGenerationMetrics {
    /// Total reports generated
    pub total_reports: usize,
    /// Generation success rate
    pub success_rate: f64,
    /// Average generation time
    pub average_generation_time: Duration,
    /// Resource utilization metrics
    pub resource_utilization: ResourceUtilization,
    /// Error statistics
    pub error_statistics: ErrorStatistics,
}

/// Individual generator performance metrics
///
/// Tracks performance statistics for specific report generators
/// including timing, success rates, and resource usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorPerformanceMetrics {
    /// Total reports generated
    pub reports_generated: usize,
    /// Average generation time
    pub average_generation_time: Duration,
    /// Success rate percentage
    pub success_rate: f64,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// Error counts by type
    pub error_counts: HashMap<String, usize>,
    /// Last generation timestamp
    pub last_generation: Option<DateTime<Utc>>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Average memory usage in bytes
    pub average_usage: usize,
}

/// System resource utilization metrics
///
/// Monitors CPU, memory, disk, and network usage
/// for performance analysis and capacity planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Disk I/O statistics
    pub disk_io: DiskIOStatistics,
    /// Network I/O statistics
    pub network_io: NetworkIOStatistics,
}

/// Disk I/O performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOStatistics {
    /// Read operations per second
    pub read_ops_per_sec: f64,
    /// Write operations per second
    pub write_ops_per_sec: f64,
    /// Average read latency
    pub average_read_latency: Duration,
    /// Average write latency
    pub average_write_latency: Duration,
}

/// Network I/O performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIOStatistics {
    /// Bytes received per second
    pub bytes_received_per_sec: f64,
    /// Bytes sent per second
    pub bytes_sent_per_sec: f64,
    /// Network latency
    pub network_latency: Duration,
}

/// Error statistics and analysis
///
/// Tracks error occurrences, patterns, and common issues
/// for system reliability monitoring and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Error count by type
    pub error_counts: HashMap<String, usize>,
    /// Error rate by generator
    pub error_rates: HashMap<String, f64>,
    /// Most common errors
    pub common_errors: Vec<CommonError>,
}

/// Common error tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonError {
    /// Error message
    pub message: String,
    /// Occurrence count
    pub count: usize,
    /// First occurrence
    pub first_occurrence: DateTime<Utc>,
    /// Last occurrence
    pub last_occurrence: DateTime<Utc>,
}

/// Data quality metrics
///
/// Tracks data completeness, accuracy, consistency,
/// and validity for data quality monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    /// Completeness percentage
    pub completeness: f64,
    /// Accuracy percentage
    pub accuracy: f64,
    /// Consistency percentage
    pub consistency: f64,
    /// Validity percentage
    pub validity: f64,
}

/// Execution metrics for scheduled jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Average execution time
    pub average_execution_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Job throughput
    pub throughput: f64,
}

/// Data source health monitoring system
///
/// Monitors data source availability, performance, and health
/// with configurable health checks and alerting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceHealthMonitor {
    /// Health check configuration
    pub health_checks: Vec<HealthCheck>,
    /// Health status for each data source
    pub health_status: HashMap<String, HealthStatus>,
    /// Alert configuration for health issues
    pub alert_config: HealthAlertConfig,
}

/// Individual health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Health check identifier
    pub check_id: String,
    /// Data source identifier
    pub data_source_id: String,
    /// Check interval
    pub interval: Duration,
    /// Timeout for health check
    pub timeout: Duration,
    /// Health check query or command
    pub check_command: String,
    /// Expected response
    pub expected_response: String,
}

/// Health status for data sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Healthy and operational
    Healthy,
    /// Degraded performance
    Degraded,
    /// Unhealthy or failing
    Unhealthy,
    /// Status unknown
    Unknown,
}

/// Health alert configuration
///
/// Configures alerting thresholds, delivery channels,
/// and rate limiting for health monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlertConfig {
    /// Enable health alerts
    pub enabled: bool,
    /// Alert threshold configuration
    pub thresholds: AlertThresholds,
    /// Alert delivery channels
    pub delivery_channels: Vec<String>,
    /// Alert frequency limits
    pub rate_limiting: AlertRateLimiting,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Response time threshold
    pub response_time_threshold: Duration,
    /// Error rate threshold
    pub error_rate_threshold: f64,
    /// Availability threshold
    pub availability_threshold: f64,
}

/// Alert rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRateLimiting {
    /// Maximum alerts per hour
    pub max_alerts_per_hour: usize,
    /// Minimum time between alerts
    pub min_alert_interval: Duration,
}

/// Notification settings for system events
///
/// Manages notification channels, escalation policies,
/// and message templates for different event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    pub success_notifications: Vec<NotificationChannel>,
    pub failure_notifications: Vec<NotificationChannel>,
    pub warning_notifications: Vec<NotificationChannel>,
    pub escalation_policy: EscalationPolicy,
}

/// Individual notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub channel_type: NotificationChannelType,
    pub channel_config: ChannelConfig,
    pub message_template: MessageTemplate,
}

/// Notification channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Email,
    Slack,
    Teams,
    SMS,
    Webhook,
    Custom(String),
}

/// Channel configuration for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    pub endpoint: String,
    pub authentication: Option<AuthenticationMethod>,
    pub rate_limiting: Option<RateLimiting>,
}

/// Authentication method for notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    /// No authentication required
    None,
    /// Basic username/password authentication
    Basic(String, String),
    /// Token-based authentication
    Token(String),
    /// Certificate-based authentication
    Certificate(PathBuf),
    /// OAuth2 authentication
    OAuth2(OAuth2Config),
    /// Custom authentication method
    Custom(String),
}

/// OAuth2 authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    /// OAuth2 client ID
    pub client_id: String,
    /// OAuth2 client secret
    pub client_secret: String,
    /// Authorization endpoint URL
    pub authorization_url: String,
    /// Token endpoint URL
    pub token_url: String,
    /// Required OAuth2 scopes
    pub scope: Vec<String>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiting {
    pub max_requests_per_minute: usize,
    pub burst_capacity: usize,
}

/// Message template for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageTemplate {
    pub subject_template: String,
    pub body_template: String,
    pub format: MessageFormat,
    pub variables: HashMap<String, String>,
}

/// Message format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageFormat {
    PlainText,
    HTML,
    Markdown,
    JSON,
    Custom(String),
}

/// Escalation policy for critical alerts
///
/// Defines escalation levels, timeouts, and acknowledgment
/// requirements for critical system alerts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    pub escalation_levels: Vec<EscalationLevel>,
    pub escalation_timeout: Duration,
    pub max_escalations: usize,
}

/// Individual escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    pub level: usize,
    pub notification_channels: Vec<String>,
    pub timeout: Duration,
    pub required_acknowledgment: bool,
}

impl ReportGenerationMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            total_reports: 0,
            success_rate: 0.0,
            average_generation_time: Duration::from_secs(0),
            resource_utilization: ResourceUtilization::default(),
            error_statistics: ErrorStatistics::default(),
        }
    }

    /// Update metrics with a new report generation
    pub fn record_generation(&mut self, success: bool, duration: Duration) {
        self.total_reports += 1;

        if success {
            // Update success rate
            let successful_reports = (self.success_rate * (self.total_reports - 1) as f64) as usize + 1;
            self.success_rate = successful_reports as f64 / self.total_reports as f64;
        } else {
            // Update success rate for failure
            let successful_reports = (self.success_rate * (self.total_reports - 1) as f64) as usize;
            self.success_rate = successful_reports as f64 / self.total_reports as f64;
        }

        // Update average generation time
        let total_duration = self.average_generation_time * (self.total_reports - 1) as u32 + duration;
        self.average_generation_time = total_duration / self.total_reports as u32;
    }

    /// Record an error occurrence
    pub fn record_error(&mut self, error_type: String) {
        *self.error_statistics.error_counts.entry(error_type).or_insert(0) += 1;
    }

    /// Get current metrics snapshot
    pub fn get_snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            timestamp: Utc::now(),
            total_reports: self.total_reports,
            success_rate: self.success_rate,
            average_generation_time: self.average_generation_time,
            error_count: self.error_statistics.error_counts.values().sum(),
        }
    }
}

/// Metrics snapshot for point-in-time analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_reports: usize,
    pub success_rate: f64,
    pub average_generation_time: Duration,
    pub error_count: usize,
}

impl DataSourceHealthMonitor {
    /// Create a new health monitor
    pub fn new() -> Self {
        Self {
            health_checks: vec![],
            health_status: HashMap::new(),
            alert_config: HealthAlertConfig::default(),
        }
    }

    /// Add a health check
    pub fn add_health_check(&mut self, check: HealthCheck) {
        self.health_checks.push(check);
    }

    /// Update health status for a data source
    pub fn update_health_status(&mut self, data_source_id: String, status: HealthStatus) {
        self.health_status.insert(data_source_id, status);
    }

    /// Get health status for a data source
    pub fn get_health_status(&self, data_source_id: &str) -> Option<&HealthStatus> {
        self.health_status.get(data_source_id)
    }

    /// Get all unhealthy data sources
    pub fn get_unhealthy_sources(&self) -> Vec<(&String, &HealthStatus)> {
        self.health_status
            .iter()
            .filter(|(_, status)| matches!(status, HealthStatus::Unhealthy | HealthStatus::Degraded))
            .collect()
    }
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            disk_io: DiskIOStatistics::default(),
            network_io: NetworkIOStatistics::default(),
        }
    }
}

impl Default for DiskIOStatistics {
    fn default() -> Self {
        Self {
            read_ops_per_sec: 0.0,
            write_ops_per_sec: 0.0,
            average_read_latency: Duration::from_millis(0),
            average_write_latency: Duration::from_millis(0),
        }
    }
}

impl Default for NetworkIOStatistics {
    fn default() -> Self {
        Self {
            bytes_received_per_sec: 0.0,
            bytes_sent_per_sec: 0.0,
            network_latency: Duration::from_millis(0),
        }
    }
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            error_counts: HashMap::new(),
            error_rates: HashMap::new(),
            common_errors: vec![],
        }
    }
}

impl Default for DataQualityMetrics {
    fn default() -> Self {
        Self {
            completeness: 100.0,
            accuracy: 100.0,
            consistency: 100.0,
            validity: 100.0,
        }
    }
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            average_execution_time: Duration::default(),
            success_rate: 0.0,
            throughput: 0.0,
        }
    }
}

impl Default for HealthAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: AlertThresholds::default(),
            delivery_channels: vec![],
            rate_limiting: AlertRateLimiting::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            response_time_threshold: Duration::from_secs(5),
            error_rate_threshold: 0.05,
            availability_threshold: 0.99,
        }
    }
}

impl Default for AlertRateLimiting {
    fn default() -> Self {
        Self {
            max_alerts_per_hour: 10,
            min_alert_interval: Duration::from_secs(300),
        }
    }
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            success_notifications: Vec::new(),
            failure_notifications: Vec::new(),
            warning_notifications: Vec::new(),
            escalation_policy: EscalationPolicy::default(),
        }
    }
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            escalation_levels: Vec::new(),
            escalation_timeout: Duration::from_secs(1800), // 30 minutes
            max_escalations: 3,
        }
    }
}