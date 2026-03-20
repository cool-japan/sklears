//! Metrics collection and monitoring systems
//!
//! This module provides comprehensive metrics collection and monitoring capabilities including:
//! - Performance metrics tracking with detailed measurements
//! - Error metrics and recovery analytics
//! - Usage analytics and engagement tracking
//! - Resource utilization monitoring
//! - Real-time metrics aggregation and reporting

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

/// Comprehensive chart rendering metrics collection system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartRenderingMetrics {
    /// Basic rendering statistics
    pub rendering_stats: RenderingStatistics,
    /// Performance metrics and measurements
    pub performance_metrics: PerformanceMetrics,
    /// Error tracking and recovery metrics
    pub error_metrics: ErrorMetrics,
    /// Usage patterns and analytics
    pub usage_metrics: UsageMetrics,
}

/// Basic rendering statistics for operational monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingStatistics {
    /// Total number of render operations attempted
    pub total_renders: u64,
    /// Number of successful render operations
    pub successful_renders: u64,
    /// Number of failed render operations
    pub failed_renders: u64,
    /// Average time taken for render operations
    pub average_render_time: Duration,
    /// Distribution of render times across buckets
    pub render_time_distribution: HashMap<String, u64>,
}

/// Comprehensive performance metrics for system optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Throughput measurements
    pub throughput: ThroughputMetrics,
    /// Latency measurements and percentiles
    pub latency: LatencyMetrics,
    /// Resource utilization across system components
    pub resource_utilization: ResourceUtilizationMetrics,
    /// Quality metrics and user experience scores
    pub quality_metrics: QualityMetrics,
}

/// Throughput metrics for capacity planning and scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Render operations completed per second
    pub renders_per_second: f64,
    /// Chart generation rate per minute
    pub charts_per_minute: f64,
    /// Data points processed per second
    pub data_points_per_second: f64,
    /// Peak throughput achieved during measurement period
    pub peak_throughput: f64,
}

/// Latency metrics for performance analysis and SLA monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Mean latency across all operations
    pub average_latency: Duration,
    /// 50th percentile latency
    pub median_latency: Duration,
    /// 95th percentile latency for SLA tracking
    pub p95_latency: Duration,
    /// 99th percentile latency for outlier analysis
    pub p99_latency: Duration,
}

/// Resource utilization metrics for capacity monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    /// CPU utilization percentage (0.0 to 100.0)
    pub cpu_utilization: f64,
    /// Memory utilization percentage (0.0 to 100.0)
    pub memory_utilization: f64,
    /// GPU utilization percentage (0.0 to 100.0)
    pub gpu_utilization: f64,
    /// Network utilization percentage (0.0 to 100.0)
    pub network_utilization: f64,
}

/// Quality metrics for user experience and output assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Visual quality score (0.0 to 100.0)
    pub visual_quality: f64,
    /// Data accuracy score (0.0 to 100.0)
    pub accuracy: f64,
    /// Output consistency score (0.0 to 100.0)
    pub consistency: f64,
    /// User satisfaction score (0.0 to 100.0)
    pub user_satisfaction: f64,
}

/// Error metrics for reliability tracking and incident management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Overall error rate percentage
    pub error_rate: f64,
    /// Distribution of errors by type or category
    pub error_distribution: HashMap<String, u64>,
    /// Recovery success rate percentage
    pub recovery_rate: f64,
    /// Average time to recover from errors
    pub mean_time_to_recovery: Duration,
}

/// Usage metrics for analytics and product insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetrics {
    /// Usage frequency by chart type
    pub chart_type_usage: HashMap<String, u64>,
    /// Feature utilization tracking
    pub feature_usage: HashMap<String, u64>,
    /// User engagement and interaction metrics
    pub engagement_metrics: EngagementMetrics,
    /// Geographic usage distribution
    pub geographic_distribution: HashMap<String, u64>,
}

/// User engagement metrics for product analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementMetrics {
    /// Average session duration per user
    pub session_duration: Duration,
    /// Total interactions per session
    pub interaction_count: u64,
    /// Percentage of users who return
    pub return_user_rate: f64,
    /// Rate of feature adoption by users
    pub feature_adoption_rate: f64,
}

/// Resource monitoring configuration for system oversight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoring {
    /// Monitoring check interval
    pub interval: Duration,
    /// List of resources to monitor
    pub monitored_resources: Vec<ResourceType>,
    /// Metrics to collect during monitoring
    pub metrics: Vec<MonitoringMetric>,
    /// Alert threshold configurations
    pub alert_thresholds: HashMap<String, f64>,
}

/// Types of resources available for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU processing resources
    CPU,
    /// System memory resources
    Memory,
    /// Graphics processing unit resources
    GPU,
    /// Network bandwidth resources
    Network,
    /// Storage I/O resources
    Storage,
}

/// Monitoring metric types for data collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringMetric {
    /// Resource utilization percentage
    Utilization,
    /// System availability metrics
    Availability,
    /// Response time measurements
    Latency,
    /// Processing throughput metrics
    Throughput,
    /// Error rate tracking
    ErrorRate,
    /// Custom metric definition
    Custom(String),
}

/// Metrics aggregation configuration for data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAggregation {
    /// Enable metrics aggregation
    pub enabled: bool,
    /// Aggregation time window
    pub window_size: Duration,
    /// Aggregation functions to apply
    pub aggregation_functions: Vec<AggregationFunction>,
    /// Data retention policy
    pub retention_policy: RetentionPolicy,
}

/// Available aggregation functions for metrics processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    /// Average value calculation
    Average,
    /// Sum total calculation
    Sum,
    /// Minimum value tracking
    Min,
    /// Maximum value tracking
    Max,
    /// Count of observations
    Count,
    /// Percentile calculations
    Percentile(f64),
    /// Standard deviation calculation
    StandardDeviation,
    /// Custom aggregation function
    Custom(String),
}

/// Data retention policy for metrics storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Retention duration for raw metrics
    pub raw_data_retention: Duration,
    /// Retention duration for aggregated metrics
    pub aggregated_data_retention: Duration,
    /// Compression settings for old data
    pub compression_enabled: bool,
    /// Archive settings for historical data
    pub archive_settings: ArchiveSettings,
}

/// Archive settings for long-term data storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveSettings {
    /// Enable archiving of old metrics
    pub enabled: bool,
    /// Archive storage location
    pub storage_location: String,
    /// Archive compression level
    pub compression_level: u8,
    /// Archive access frequency
    pub access_frequency: AccessFrequency,
}

/// Archive access frequency classifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessFrequency {
    /// Frequently accessed data
    Frequent,
    /// Occasionally accessed data
    Occasional,
    /// Rarely accessed data
    Rare,
    /// Archive-only data
    Archive,
}

/// Real-time metrics streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetrics {
    /// Enable real-time streaming
    pub enabled: bool,
    /// Streaming update interval
    pub update_interval: Duration,
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Streaming protocols supported
    pub protocols: Vec<StreamingProtocol>,
}

/// Streaming protocols for real-time data delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingProtocol {
    /// WebSocket streaming
    WebSocket,
    /// Server-sent events
    SSE,
    /// HTTP polling
    HTTP,
    /// Custom streaming protocol
    Custom(String),
}

/// Metrics alert system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAlertSystem {
    /// Enable alerting system
    pub enabled: bool,
    /// Alert rules and conditions
    pub alert_rules: Vec<AlertRule>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
    /// Alert escalation policy
    pub escalation_policy: AlertEscalationPolicy,
}

/// Individual alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name and description
    pub rule_name: String,
    /// Metric to monitor
    pub metric_name: String,
    /// Alert condition
    pub condition: AlertCondition,
    /// Alert severity level
    pub severity: AlertSeverity,
}

/// Alert condition types for rule evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Threshold exceeded
    Threshold(f64),
    /// Rate of change exceeded
    RateOfChange(f64),
    /// Anomaly detection triggered
    Anomaly,
    /// Custom condition logic
    Custom(String),
}

/// Alert severity levels for prioritization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Low priority alert
    Low,
    /// Medium priority alert
    Medium,
    /// High priority alert
    High,
    /// Critical priority alert
    Critical,
}

/// Notification channels for alert delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notifications
    Email(String),
    /// SMS notifications
    SMS(String),
    /// Slack notifications
    Slack(String),
    /// Webhook notifications
    Webhook(String),
    /// Custom notification channel
    Custom(String),
}

/// Alert escalation policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEscalationPolicy {
    /// Escalation levels and timing
    pub escalation_levels: Vec<EscalationLevel>,
    /// Maximum escalation timeout
    pub max_escalation_time: Duration,
}

/// Individual escalation level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level identifier
    pub level_id: String,
    /// Time to wait before escalating
    pub wait_time: Duration,
    /// Notification channels for this level
    pub channels: Vec<NotificationChannel>,
}

impl Default for ChartRenderingMetrics {
    fn default() -> Self {
        Self {
            rendering_stats: RenderingStatistics::default(),
            performance_metrics: PerformanceMetrics::default(),
            error_metrics: ErrorMetrics::default(),
            usage_metrics: UsageMetrics::default(),
        }
    }
}

impl Default for RenderingStatistics {
    fn default() -> Self {
        Self {
            total_renders: 0,
            successful_renders: 0,
            failed_renders: 0,
            average_render_time: Duration::from_secs(0),
            render_time_distribution: HashMap::new(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: ThroughputMetrics::default(),
            latency: LatencyMetrics::default(),
            resource_utilization: ResourceUtilizationMetrics::default(),
            quality_metrics: QualityMetrics::default(),
        }
    }
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            renders_per_second: 0.0,
            charts_per_minute: 0.0,
            data_points_per_second: 0.0,
            peak_throughput: 0.0,
        }
    }
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            average_latency: Duration::from_secs(0),
            median_latency: Duration::from_secs(0),
            p95_latency: Duration::from_secs(0),
            p99_latency: Duration::from_secs(0),
        }
    }
}

impl Default for ResourceUtilizationMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: 0.0,
            network_utilization: 0.0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            visual_quality: 100.0,
            accuracy: 100.0,
            consistency: 100.0,
            user_satisfaction: 100.0,
        }
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            error_rate: 0.0,
            error_distribution: HashMap::new(),
            recovery_rate: 100.0,
            mean_time_to_recovery: Duration::from_secs(0),
        }
    }
}

impl Default for UsageMetrics {
    fn default() -> Self {
        Self {
            chart_type_usage: HashMap::new(),
            feature_usage: HashMap::new(),
            engagement_metrics: EngagementMetrics::default(),
            geographic_distribution: HashMap::new(),
        }
    }
}

impl Default for EngagementMetrics {
    fn default() -> Self {
        Self {
            session_duration: Duration::from_secs(0),
            interaction_count: 0,
            return_user_rate: 0.0,
            feature_adoption_rate: 0.0,
        }
    }
}

impl Default for ResourceMonitoring {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            monitored_resources: vec![ResourceType::CPU, ResourceType::Memory],
            metrics: vec![MonitoringMetric::Utilization],
            alert_thresholds: HashMap::new(),
        }
    }
}