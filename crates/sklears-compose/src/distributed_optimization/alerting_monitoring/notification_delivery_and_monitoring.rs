use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::notification_message_types::{PendingNotification, NotificationPriority, DeliveryPerformanceMetrics};

/// Delivery tracker for monitoring notification delivery
#[derive(Debug, Clone)]
pub struct DeliveryTracker {
    /// Active deliveries
    pub active_deliveries: HashMap<String, ActiveDelivery>,
    /// Delivery history
    pub delivery_history: VecDeque<DeliveryRecord>,
    /// Tracking configuration
    pub config: DeliveryTrackerConfig,
    /// Performance statistics
    pub performance_stats: DeliveryPerformanceStats,
}

/// Active delivery tracking
#[derive(Debug, Clone)]
pub struct ActiveDelivery {
    /// Delivery ID
    pub delivery_id: String,
    /// Notification ID
    pub notification_id: String,
    /// Channel ID
    pub channel_id: String,
    /// Start time
    pub start_time: SystemTime,
    /// Current status
    pub status: DeliveryStatus,
    /// Progress information
    pub progress: DeliveryProgress,
    /// Performance metrics
    pub metrics: DeliveryMetrics,
}

/// Delivery status
#[derive(Debug, Clone)]
pub enum DeliveryStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Timeout,
}

/// Delivery progress information
#[derive(Debug, Clone)]
pub struct DeliveryProgress {
    /// Current step
    pub current_step: DeliveryStep,
    /// Total steps
    pub total_steps: u32,
    /// Completed steps
    pub completed_steps: u32,
    /// Progress percentage
    pub progress_percentage: f64,
    /// Estimated completion time
    pub estimated_completion: Option<SystemTime>,
}

/// Delivery steps
#[derive(Debug, Clone)]
pub enum DeliveryStep {
    Queued,
    Formatting,
    Authenticating,
    Connecting,
    Sending,
    AwaitingResponse,
    Processing,
    Completed,
}

/// Delivery metrics
#[derive(Debug, Clone)]
pub struct DeliveryMetrics {
    /// Start timestamp
    pub start_timestamp: SystemTime,
    /// End timestamp
    pub end_timestamp: Option<SystemTime>,
    /// Duration
    pub duration: Option<Duration>,
    /// Bytes sent
    pub bytes_sent: usize,
    /// Response size
    pub response_size: Option<usize>,
    /// Error count
    pub error_count: u32,
}

/// Delivery record for history
#[derive(Debug, Clone)]
pub struct DeliveryRecord {
    /// Record ID
    pub record_id: String,
    /// Delivery ID
    pub delivery_id: String,
    /// Notification ID
    pub notification_id: String,
    /// Channel ID
    pub channel_id: String,
    /// Final status
    pub final_status: DeliveryStatus,
    /// Start time
    pub start_time: SystemTime,
    /// End time
    pub end_time: SystemTime,
    /// Total duration
    pub duration: Duration,
    /// Attempt count
    pub attempt_count: u32,
    /// Error message
    pub error_message: Option<String>,
    /// Performance metrics
    pub performance_metrics: DeliveryPerformanceMetrics,
}

/// Delivery tracker configuration
#[derive(Debug, Clone)]
pub struct DeliveryTrackerConfig {
    /// Maximum active deliveries to track
    pub max_active_deliveries: usize,
    /// History retention size
    pub history_retention_size: usize,
    /// History retention time
    pub history_retention_time: Duration,
    /// Enable detailed tracking
    pub enable_detailed_tracking: bool,
    /// Performance monitoring interval
    pub performance_monitoring_interval: Duration,
}

/// Delivery performance statistics
#[derive(Debug, Clone)]
pub struct DeliveryPerformanceStats {
    /// Total deliveries
    pub total_deliveries: u64,
    /// Successful deliveries
    pub successful_deliveries: u64,
    /// Failed deliveries
    pub failed_deliveries: u64,
    /// Average delivery time
    pub average_delivery_time: Duration,
    /// Delivery rate (per minute)
    pub delivery_rate: f64,
    /// Success rate percentage
    pub success_rate: f64,
    /// Channel performance
    pub channel_performance: HashMap<String, ChannelPerformanceStats>,
}

/// Performance statistics per channel
#[derive(Debug, Clone)]
pub struct ChannelPerformanceStats {
    /// Channel deliveries
    pub deliveries: u64,
    /// Channel success rate
    pub success_rate: f64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Error rate
    pub error_rate: f64,
    /// Last successful delivery
    pub last_successful_delivery: Option<SystemTime>,
}

/// Comprehensive notification metrics
#[derive(Debug, Clone)]
pub struct NotificationMetrics {
    /// Overall system metrics
    pub system_metrics: SystemNotificationMetrics,
    /// Channel metrics
    pub channel_metrics: HashMap<String, ChannelNotificationMetrics>,
    /// Performance metrics
    pub performance_metrics: NotificationPerformanceMetrics,
    /// Error metrics
    pub error_metrics: NotificationErrorMetrics,
}

/// System-wide notification metrics
#[derive(Debug, Clone)]
pub struct SystemNotificationMetrics {
    /// Total notifications sent
    pub total_sent: u64,
    /// Notifications sent today
    pub sent_today: u64,
    /// Success rate
    pub success_rate: f64,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Queue length
    pub current_queue_length: usize,
    /// System throughput
    pub throughput: f64,
}

/// Channel-specific notification metrics
#[derive(Debug, Clone)]
pub struct ChannelNotificationMetrics {
    /// Channel ID
    pub channel_id: String,
    /// Messages sent
    pub messages_sent: u64,
    /// Success count
    pub success_count: u64,
    /// Failure count
    pub failure_count: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Channel health score
    pub health_score: f64,
}

/// Performance metrics for notifications
#[derive(Debug, Clone)]
pub struct NotificationPerformanceMetrics {
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Latency metrics
    pub latency: LatencyMetrics,
    /// Resource utilization
    pub resource_utilization: ResourceUtilizationMetrics,
    /// Scaling metrics
    pub scaling_metrics: ScalingMetrics,
}

/// Throughput metrics
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Messages per second
    pub messages_per_second: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Sustained throughput
    pub sustained_throughput: f64,
    /// Throughput trend
    pub throughput_trend: ThroughputTrend,
}

/// Throughput trend analysis
#[derive(Debug, Clone)]
pub enum ThroughputTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Latency metrics
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    /// Average latency
    pub average_latency: Duration,
    /// Median latency
    pub median_latency: Duration,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// 99th percentile latency
    pub p99_latency: Duration,
    /// Maximum latency
    pub max_latency: Duration,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilizationMetrics {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Storage utilization
    pub storage_utilization: f64,
}

/// Scaling metrics
#[derive(Debug, Clone)]
pub struct ScalingMetrics {
    /// Current capacity
    pub current_capacity: usize,
    /// Maximum capacity
    pub max_capacity: usize,
    /// Utilization percentage
    pub utilization_percentage: f64,
    /// Scaling recommendations
    pub scaling_recommendations: Vec<ScalingRecommendation>,
}

/// Scaling recommendation
#[derive(Debug, Clone)]
pub struct ScalingRecommendation {
    /// Recommendation type
    pub recommendation_type: ScalingRecommendationType,
    /// Recommended action
    pub recommended_action: String,
    /// Confidence score
    pub confidence_score: f64,
    /// Expected impact
    pub expected_impact: String,
}

/// Scaling recommendation types
#[derive(Debug, Clone)]
pub enum ScalingRecommendationType {
    ScaleUp,
    ScaleDown,
    ScaleOut,
    ScaleIn,
    Optimize,
    Maintain,
}

/// Error metrics for notifications
#[derive(Debug, Clone)]
pub struct NotificationErrorMetrics {
    /// Total errors
    pub total_errors: u64,
    /// Error rate
    pub error_rate: f64,
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
    /// Errors by channel
    pub errors_by_channel: HashMap<String, u64>,
    /// Recent error trend
    pub error_trend: ErrorTrendMetrics,
}

/// Error trend metrics
#[derive(Debug, Clone)]
pub struct ErrorTrendMetrics {
    /// Error count in last hour
    pub last_hour_errors: u64,
    /// Error count in last day
    pub last_day_errors: u64,
    /// Error rate trend
    pub error_rate_trend: ErrorRateTrend,
    /// Most common error types
    pub common_error_types: Vec<(String, u64)>,
}

/// Error rate trend
#[derive(Debug, Clone)]
pub enum ErrorRateTrend {
    Improving,
    Worsening,
    Stable,
    Volatile,
}

/// Channel health monitor for tracking channel status
#[derive(Debug, Clone)]
pub struct ChannelHealthMonitor {
    /// Channel health status
    pub channel_health: HashMap<String, ChannelHealth>,
    /// Health monitoring configuration
    pub config: HealthMonitorConfig,
    /// Health check scheduler
    pub scheduler: HealthCheckScheduler,
    /// Health statistics
    pub statistics: HealthStatistics,
}

/// Channel health status
#[derive(Debug, Clone)]
pub struct ChannelHealth {
    /// Channel ID
    pub channel_id: String,
    /// Health status
    pub status: HealthStatus,
    /// Health score (0-100)
    pub health_score: f64,
    /// Last health check
    pub last_check: SystemTime,
    /// Health history
    pub health_history: VecDeque<HealthCheckResult>,
    /// Current issues
    pub current_issues: Vec<HealthIssue>,
}

/// Health status
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
    Maintenance,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Check timestamp
    pub timestamp: SystemTime,
    /// Check success
    pub success: bool,
    /// Response time
    pub response_time: Duration,
    /// Health score
    pub health_score: f64,
    /// Issues found
    pub issues: Vec<HealthIssue>,
}

/// Health issue
#[derive(Debug, Clone)]
pub struct HealthIssue {
    /// Issue type
    pub issue_type: HealthIssueType,
    /// Issue severity
    pub severity: HealthIssueSeverity,
    /// Issue description
    pub description: String,
    /// Issue timestamp
    pub timestamp: SystemTime,
    /// Issue metadata
    pub metadata: HashMap<String, String>,
}

/// Health issue types
#[derive(Debug, Clone)]
pub enum HealthIssueType {
    Connectivity,
    Authentication,
    RateLimit,
    Performance,
    Configuration,
    Custom(String),
}

/// Health issue severity
#[derive(Debug, Clone)]
pub enum HealthIssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Health monitor configuration
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Default check interval
    pub default_check_interval: Duration,
    /// Check timeout
    pub check_timeout: Duration,
    /// Health history size
    pub health_history_size: usize,
    /// Enable automatic remediation
    pub enable_auto_remediation: bool,
    /// Remediation strategies
    pub remediation_strategies: Vec<RemediationStrategy>,
}

/// Remediation strategies
#[derive(Debug, Clone)]
pub struct RemediationStrategy {
    /// Strategy name
    pub name: String,
    /// Trigger conditions
    pub trigger_conditions: Vec<RemediationTrigger>,
    /// Actions to take
    pub actions: Vec<RemediationAction>,
    /// Strategy priority
    pub priority: u32,
}

/// Remediation trigger conditions
#[derive(Debug, Clone)]
pub struct RemediationTrigger {
    /// Trigger type
    pub trigger_type: RemediationTriggerType,
    /// Threshold value
    pub threshold: f64,
    /// Condition operator
    pub operator: ConditionOperator,
    /// Duration for trigger
    pub duration: Duration,
}

/// Remediation trigger types
#[derive(Debug, Clone)]
pub enum RemediationTriggerType {
    ErrorRate,
    ResponseTime,
    HealthScore,
    Availability,
    Custom(String),
}

/// Condition operators
#[derive(Debug, Clone)]
pub enum ConditionOperator {
    GreaterThan,
    LessThan,
    Equals,
    GreaterThanOrEquals,
    LessThanOrEquals,
    NotEquals,
}

/// Remediation actions
#[derive(Debug, Clone)]
pub struct RemediationAction {
    /// Action type
    pub action_type: RemediationActionType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Action timeout
    pub timeout: Duration,
    /// Retry configuration
    pub retry_config: Option<ActionRetryConfig>,
}

/// Remediation action types
#[derive(Debug, Clone)]
pub enum RemediationActionType {
    RestartChannel,
    RefreshCredentials,
    ClearCache,
    ScaleResources,
    NotifyOperators,
    Custom(String),
}

/// Action retry configuration
#[derive(Debug, Clone)]
pub struct ActionRetryConfig {
    /// Maximum retries
    pub max_retries: u32,
    /// Retry delay
    pub retry_delay: Duration,
    /// Exponential backoff
    pub exponential_backoff: bool,
}

/// Health check scheduler
#[derive(Debug, Clone)]
pub struct HealthCheckScheduler {
    /// Scheduled checks
    pub scheduled_checks: HashMap<String, ScheduledHealthCheck>,
    /// Scheduler configuration
    pub config: SchedulerConfig,
    /// Scheduler statistics
    pub statistics: SchedulerStatistics,
}

/// Scheduled health check
#[derive(Debug, Clone)]
pub struct ScheduledHealthCheck {
    /// Channel ID
    pub channel_id: String,
    /// Check interval
    pub check_interval: Duration,
    /// Next check time
    pub next_check_time: SystemTime,
    /// Check priority
    pub priority: HealthCheckPriority,
    /// Check configuration
    pub check_config: HealthCheckConfig,
}

/// Health check priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum HealthCheckPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Check type
    pub check_type: HealthCheckType,
    /// Check parameters
    pub parameters: HashMap<String, String>,
    /// Expected response time
    pub expected_response_time: Duration,
    /// Success criteria
    pub success_criteria: Vec<SuccessCriteria>,
}

/// Health check types
#[derive(Debug, Clone)]
pub enum HealthCheckType {
    Ping,
    Echo,
    TestMessage,
    Authentication,
    Custom(String),
}

/// Success criteria for health checks
#[derive(Debug, Clone)]
pub struct SuccessCriteria {
    /// Criteria type
    pub criteria_type: CriteriaType,
    /// Expected value
    pub expected_value: String,
    /// Tolerance
    pub tolerance: Option<f64>,
}

/// Success criteria types
#[derive(Debug, Clone)]
pub enum CriteriaType {
    ResponseTime,
    StatusCode,
    ResponseContent,
    ConnectionSuccess,
    Custom(String),
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum concurrent checks
    pub max_concurrent_checks: usize,
    /// Default check timeout
    pub default_timeout: Duration,
    /// Enable priority scheduling
    pub enable_priority_scheduling: bool,
    /// Check result retention
    pub result_retention_duration: Duration,
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStatistics {
    /// Total checks scheduled
    pub total_scheduled: u64,
    /// Total checks completed
    pub total_completed: u64,
    /// Average check duration
    pub avg_check_duration: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Active checks
    pub active_checks: usize,
}

/// Health statistics
#[derive(Debug, Clone)]
pub struct HealthStatistics {
    /// Overall system health
    pub overall_health_score: f64,
    /// Healthy channels count
    pub healthy_channels: usize,
    /// Warning channels count
    pub warning_channels: usize,
    /// Critical channels count
    pub critical_channels: usize,
    /// Average health score
    pub average_health_score: f64,
    /// Health trend
    pub health_trend: HealthTrend,
}

/// Health trend analysis
#[derive(Debug, Clone)]
pub enum HealthTrend {
    Improving,
    Declining,
    Stable,
    Fluctuating,
}

/// System statistics
#[derive(Debug, Clone)]
pub struct NotificationSystemStatistics {
    /// Number of active channels
    pub active_channels: usize,
    /// Current queue length
    pub queue_length: usize,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

impl DeliveryTracker {
    /// Create a new delivery tracker
    pub fn new() -> Self {
        Self {
            active_deliveries: HashMap::new(),
            delivery_history: VecDeque::new(),
            config: DeliveryTrackerConfig::default(),
            performance_stats: DeliveryPerformanceStats::default(),
        }
    }

    /// Start tracking a delivery
    pub fn start_delivery(&mut self, delivery_id: String, notification: &PendingNotification) {
        let active_delivery = ActiveDelivery {
            delivery_id: delivery_id.clone(),
            notification_id: notification.notification_id.clone(),
            channel_id: notification.channel_id.clone(),
            start_time: SystemTime::now(),
            status: DeliveryStatus::Pending,
            progress: DeliveryProgress::new(),
            metrics: DeliveryMetrics::new(),
        };

        self.active_deliveries.insert(delivery_id, active_delivery);
    }

    /// Update delivery status
    pub fn update_delivery_status(&mut self, delivery_id: &str, status: DeliveryStatus) {
        if let Some(delivery) = self.active_deliveries.get_mut(delivery_id) {
            delivery.status = status;
            delivery.progress.update_from_status(&delivery.status);
        }
    }

    /// Complete a delivery
    pub fn complete_delivery(&mut self, delivery_id: &str, success: bool, error_message: Option<String>) {
        if let Some(mut delivery) = self.active_deliveries.remove(delivery_id) {
            let end_time = SystemTime::now();
            delivery.metrics.end_timestamp = Some(end_time);
            delivery.metrics.duration = delivery.start_time.elapsed().ok();

            let status = if success {
                DeliveryStatus::Completed
            } else {
                DeliveryStatus::Failed
            };

            let record = DeliveryRecord {
                record_id: uuid::Uuid::new_v4().to_string(),
                delivery_id: delivery.delivery_id.clone(),
                notification_id: delivery.notification_id.clone(),
                channel_id: delivery.channel_id.clone(),
                final_status: status,
                start_time: delivery.start_time,
                end_time,
                duration: delivery.metrics.duration.unwrap_or(Duration::from_secs(0)),
                attempt_count: 1,
                error_message,
                performance_metrics: DeliveryPerformanceMetrics {
                    queue_time: Duration::from_secs(0),
                    processing_time: delivery.metrics.duration.unwrap_or(Duration::from_secs(0)),
                    network_time: Duration::from_secs(0),
                    total_delivery_time: delivery.metrics.duration.unwrap_or(Duration::from_secs(0)),
                    retry_count: 0,
                },
            };

            self.add_to_history(record);
            self.update_performance_stats(success, delivery.metrics.duration.unwrap_or(Duration::from_secs(0)));
        }
    }

    fn add_to_history(&mut self, record: DeliveryRecord) {
        if self.delivery_history.len() >= self.config.history_retention_size {
            self.delivery_history.pop_front();
        }
        self.delivery_history.push_back(record);
    }

    fn update_performance_stats(&mut self, success: bool, duration: Duration) {
        self.performance_stats.total_deliveries += 1;

        if success {
            self.performance_stats.successful_deliveries += 1;
        } else {
            self.performance_stats.failed_deliveries += 1;
        }

        // Update average delivery time
        let total_time = self.performance_stats.average_delivery_time.as_nanos() *
            (self.performance_stats.total_deliveries - 1) as u128 + duration.as_nanos();
        self.performance_stats.average_delivery_time = Duration::from_nanos(
            (total_time / self.performance_stats.total_deliveries as u128) as u64
        );

        // Update success rate
        self.performance_stats.success_rate =
            (self.performance_stats.successful_deliveries as f64 /
             self.performance_stats.total_deliveries as f64) * 100.0;
    }

    /// Get active delivery count
    pub fn get_active_delivery_count(&self) -> usize {
        self.active_deliveries.len()
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &DeliveryPerformanceStats {
        &self.performance_stats
    }

    /// Get delivery history
    pub fn get_delivery_history(&self, limit: Option<usize>) -> Vec<&DeliveryRecord> {
        let limit = limit.unwrap_or(self.delivery_history.len());
        self.delivery_history.iter().rev().take(limit).collect()
    }
}

impl DeliveryProgress {
    /// Create new delivery progress
    pub fn new() -> Self {
        Self {
            current_step: DeliveryStep::Queued,
            total_steps: 8,
            completed_steps: 0,
            progress_percentage: 0.0,
            estimated_completion: None,
        }
    }

    /// Update progress from delivery status
    pub fn update_from_status(&mut self, status: &DeliveryStatus) {
        match status {
            DeliveryStatus::Pending => {
                self.current_step = DeliveryStep::Queued;
                self.completed_steps = 0;
            }
            DeliveryStatus::InProgress => {
                self.current_step = DeliveryStep::Sending;
                self.completed_steps = 4;
            }
            DeliveryStatus::Completed => {
                self.current_step = DeliveryStep::Completed;
                self.completed_steps = self.total_steps;
            }
            _ => {}
        }

        self.progress_percentage = (self.completed_steps as f64 / self.total_steps as f64) * 100.0;
    }
}

impl DeliveryMetrics {
    /// Create new delivery metrics
    pub fn new() -> Self {
        Self {
            start_timestamp: SystemTime::now(),
            end_timestamp: None,
            duration: None,
            bytes_sent: 0,
            response_size: None,
            error_count: 0,
        }
    }
}

impl ChannelHealthMonitor {
    /// Create a new health monitor
    pub fn new() -> Self {
        Self {
            channel_health: HashMap::new(),
            config: HealthMonitorConfig::default(),
            scheduler: HealthCheckScheduler::default(),
            statistics: HealthStatistics::default(),
        }
    }

    /// Add channel for monitoring
    pub fn add_channel(&mut self, channel_id: String) {
        let health = ChannelHealth {
            channel_id: channel_id.clone(),
            status: HealthStatus::Unknown,
            health_score: 0.0,
            last_check: SystemTime::now(),
            health_history: VecDeque::new(),
            current_issues: Vec::new(),
        };

        self.channel_health.insert(channel_id.clone(), health);

        // Schedule health check
        let scheduled_check = ScheduledHealthCheck {
            channel_id: channel_id.clone(),
            check_interval: self.config.default_check_interval,
            next_check_time: SystemTime::now() + self.config.default_check_interval,
            priority: HealthCheckPriority::Normal,
            check_config: HealthCheckConfig::default(),
        };

        self.scheduler.scheduled_checks.insert(channel_id, scheduled_check);
    }

    /// Perform health check for channel
    pub fn perform_health_check(&mut self, channel_id: &str) -> Option<HealthCheckResult> {
        if let Some(health) = self.channel_health.get_mut(channel_id) {
            let start_time = SystemTime::now();

            // Simulate health check (in real implementation, this would test the actual channel)
            let success = true; // Placeholder
            let response_time = Duration::from_millis(100); // Placeholder
            let health_score = if success { 95.0 } else { 30.0 };

            let result = HealthCheckResult {
                timestamp: start_time,
                success,
                response_time,
                health_score,
                issues: Vec::new(),
            };

            // Update health
            health.last_check = start_time;
            health.health_score = health_score;
            health.status = if success {
                HealthStatus::Healthy
            } else {
                HealthStatus::Critical
            };

            // Add to history
            if health.health_history.len() >= self.config.health_history_size {
                health.health_history.pop_front();
            }
            health.health_history.push_back(result.clone());

            // Update statistics
            self.scheduler.statistics.total_completed += 1;
            self.update_health_statistics();

            Some(result)
        } else {
            None
        }
    }

    /// Get overall health status
    pub fn get_overall_health(&self) -> f64 {
        if self.channel_health.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.channel_health.values()
            .map(|health| health.health_score)
            .sum();

        total_score / self.channel_health.len() as f64
    }

    /// Get channels needing attention
    pub fn get_unhealthy_channels(&self) -> Vec<&ChannelHealth> {
        self.channel_health.values()
            .filter(|health| matches!(health.status, HealthStatus::Warning | HealthStatus::Critical))
            .collect()
    }

    fn update_health_statistics(&mut self) {
        let mut healthy_count = 0;
        let mut warning_count = 0;
        let mut critical_count = 0;
        let mut total_score = 0.0;

        for health in self.channel_health.values() {
            total_score += health.health_score;
            match health.status {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Warning => warning_count += 1,
                HealthStatus::Critical => critical_count += 1,
                _ => {}
            }
        }

        self.statistics.healthy_channels = healthy_count;
        self.statistics.warning_channels = warning_count;
        self.statistics.critical_channels = critical_count;
        self.statistics.average_health_score = if !self.channel_health.is_empty() {
            total_score / self.channel_health.len() as f64
        } else {
            0.0
        };
        self.statistics.overall_health_score = self.get_overall_health();
    }
}

impl Default for DeliveryTrackerConfig {
    fn default() -> Self {
        Self {
            max_active_deliveries: 10000,
            history_retention_size: 50000,
            history_retention_time: Duration::from_secs(24 * 60 * 60 * 7), // 7 days
            enable_detailed_tracking: true,
            performance_monitoring_interval: Duration::from_secs(60),
        }
    }
}

impl Default for DeliveryPerformanceStats {
    fn default() -> Self {
        Self {
            total_deliveries: 0,
            successful_deliveries: 0,
            failed_deliveries: 0,
            average_delivery_time: Duration::from_secs(0),
            delivery_rate: 0.0,
            success_rate: 0.0,
            channel_performance: HashMap::new(),
        }
    }
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            default_check_interval: Duration::from_secs(300), // 5 minutes
            check_timeout: Duration::from_secs(30),
            health_history_size: 100,
            enable_auto_remediation: false,
            remediation_strategies: Vec::new(),
        }
    }
}

impl Default for HealthCheckScheduler {
    fn default() -> Self {
        Self {
            scheduled_checks: HashMap::new(),
            config: SchedulerConfig::default(),
            statistics: SchedulerStatistics::default(),
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_checks: 10,
            default_timeout: Duration::from_secs(30),
            enable_priority_scheduling: true,
            result_retention_duration: Duration::from_secs(24 * 60 * 60), // 1 day
        }
    }
}

impl Default for SchedulerStatistics {
    fn default() -> Self {
        Self {
            total_scheduled: 0,
            total_completed: 0,
            avg_check_duration: Duration::from_secs(0),
            success_rate: 0.0,
            active_checks: 0,
        }
    }
}

impl Default for HealthStatistics {
    fn default() -> Self {
        Self {
            overall_health_score: 0.0,
            healthy_channels: 0,
            warning_channels: 0,
            critical_channels: 0,
            average_health_score: 0.0,
            health_trend: HealthTrend::Stable,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_type: HealthCheckType::Ping,
            parameters: HashMap::new(),
            expected_response_time: Duration::from_secs(5),
            success_criteria: vec![
                SuccessCriteria {
                    criteria_type: CriteriaType::ResponseTime,
                    expected_value: "5000".to_string(), // 5 seconds in milliseconds
                    tolerance: Some(1000.0), // 1 second tolerance
                }
            ],
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delivery_tracker_creation() {
        let tracker = DeliveryTracker::new();
        assert_eq!(tracker.active_deliveries.len(), 0);
        assert_eq!(tracker.delivery_history.len(), 0);
        assert_eq!(tracker.performance_stats.total_deliveries, 0);
    }

    #[test]
    fn test_delivery_tracking() {
        let mut tracker = DeliveryTracker::new();

        let notification = PendingNotification::new(
            "test-123".to_string(),
            "email-channel".to_string(),
            super::super::notification_message_types::NotificationMessage {
                subject: "Test".to_string(),
                body: "Test message".to_string(),
                message_type: super::super::notification_message_types::MessageType::Alert,
                attachments: Vec::new(),
                metadata: super::super::notification_message_types::MessageMetadata {
                    source: "test".to_string(),
                    correlation_id: None,
                    thread_id: None,
                    reply_to: None,
                    tags: Vec::new(),
                    custom_fields: HashMap::new(),
                },
                rich_content: None,
                localization_data: HashMap::new(),
            },
            NotificationPriority::Normal,
        );

        let delivery_id = "delivery-123".to_string();

        // Start tracking
        tracker.start_delivery(delivery_id.clone(), &notification);
        assert_eq!(tracker.get_active_delivery_count(), 1);

        // Update status
        tracker.update_delivery_status(&delivery_id, DeliveryStatus::InProgress);

        // Complete delivery
        tracker.complete_delivery(&delivery_id, true, None);
        assert_eq!(tracker.get_active_delivery_count(), 0);
        assert_eq!(tracker.delivery_history.len(), 1);
        assert_eq!(tracker.performance_stats.total_deliveries, 1);
        assert_eq!(tracker.performance_stats.successful_deliveries, 1);
    }

    #[test]
    fn test_health_monitor() {
        let mut monitor = ChannelHealthMonitor::new();

        // Add channel
        monitor.add_channel("test-channel".to_string());
        assert!(monitor.channel_health.contains_key("test-channel"));

        // Perform health check
        let result = monitor.perform_health_check("test-channel");
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.success);
        assert!(result.health_score > 0.0);

        // Check overall health
        let overall_health = monitor.get_overall_health();
        assert!(overall_health > 0.0);
    }

    #[test]
    fn test_delivery_progress() {
        let mut progress = DeliveryProgress::new();
        assert_eq!(progress.progress_percentage, 0.0);

        progress.update_from_status(&DeliveryStatus::InProgress);
        assert!(progress.progress_percentage > 0.0);
        assert!(progress.progress_percentage < 100.0);

        progress.update_from_status(&DeliveryStatus::Completed);
        assert_eq!(progress.progress_percentage, 100.0);
    }

    #[test]
    fn test_health_statistics() {
        let mut monitor = ChannelHealthMonitor::new();

        // Add multiple channels
        monitor.add_channel("channel1".to_string());
        monitor.add_channel("channel2".to_string());
        monitor.add_channel("channel3".to_string());

        // Perform health checks
        monitor.perform_health_check("channel1");
        monitor.perform_health_check("channel2");
        monitor.perform_health_check("channel3");

        // Check statistics
        assert_eq!(monitor.statistics.healthy_channels, 3);
        assert!(monitor.statistics.average_health_score > 0.0);
    }

    #[test]
    fn test_performance_stats_calculation() {
        let mut tracker = DeliveryTracker::new();

        // Test multiple deliveries
        for i in 0..5 {
            let notification = PendingNotification::new(
                format!("test-{}", i),
                "email-channel".to_string(),
                super::super::notification_message_types::NotificationMessage {
                    subject: "Test".to_string(),
                    body: "Test message".to_string(),
                    message_type: super::super::notification_message_types::MessageType::Alert,
                    attachments: Vec::new(),
                    metadata: super::super::notification_message_types::MessageMetadata {
                        source: "test".to_string(),
                        correlation_id: None,
                        thread_id: None,
                        reply_to: None,
                        tags: Vec::new(),
                        custom_fields: HashMap::new(),
                    },
                    rich_content: None,
                    localization_data: HashMap::new(),
                },
                NotificationPriority::Normal,
            );

            let delivery_id = format!("delivery-{}", i);
            tracker.start_delivery(delivery_id.clone(), &notification);

            // Complete some successfully, some with failures
            let success = i % 2 == 0;
            tracker.complete_delivery(&delivery_id, success, None);
        }

        assert_eq!(tracker.performance_stats.total_deliveries, 5);
        assert_eq!(tracker.performance_stats.successful_deliveries, 3);
        assert_eq!(tracker.performance_stats.failed_deliveries, 2);
        assert_eq!(tracker.performance_stats.success_rate, 60.0);
    }
}