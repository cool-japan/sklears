//! Component Health Monitoring and Status Tracking
//!
//! This module provides comprehensive health monitoring capabilities for fault tolerance
//! components including health checks, status tracking, and predictive health analytics.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;

/// Component health status enumeration
///
/// Represents the various states a component can be in from a health perspective
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentHealth {
    /// Component healthy and operational
    Healthy {
        uptime: Duration,
        performance_score: f64,
    },

    /// Component degraded but functional
    Degraded {
        reason: String,
        impact_level: f64,
    },

    /// Component unhealthy but recoverable
    Unhealthy {
        error_count: usize,
        last_error: String,
    },

    /// Component failed and needs recovery
    Failed {
        failure_reason: String,
        failure_time: SystemTime,
    },

    /// Component unknown status
    Unknown {
        last_check: SystemTime,
    },
}

/// Health check configuration
///
/// Defines how health checks should be performed for a component
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: Duration,

    /// Health check timeout
    pub timeout: Duration,

    /// Health check endpoint or method
    pub endpoint: String,

    /// Expected response
    pub expected_response: Option<String>,

    /// Failure threshold before marking unhealthy
    pub failure_threshold: usize,

    /// Success threshold for recovery
    pub success_threshold: usize,

    /// Health check type
    pub check_type: HealthCheckType,

    /// Enable predictive health analysis
    pub predictive_analysis: bool,

    /// Health check metadata
    pub metadata: HashMap<String, String>,
}

/// Health check types
///
/// Different mechanisms for performing health checks
#[derive(Debug, Clone)]
pub enum HealthCheckType {
    /// HTTP endpoint check
    Http {
        method: String,
        headers: HashMap<String, String>,
        expected_status: Vec<u16>,
    },

    /// TCP port check
    Tcp {
        host: String,
        port: u16,
        connect_timeout: Duration,
    },

    /// Function call check
    Function {
        function_name: String,
        parameters: HashMap<String, String>,
    },

    /// Resource availability check
    Resource {
        resource_type: String,
        threshold: f64,
    },

    /// Memory usage check
    Memory {
        max_usage_percent: f64,
        check_virtual: bool,
    },

    /// CPU usage check
    Cpu {
        max_usage_percent: f64,
        sample_duration: Duration,
    },

    /// Disk space check
    Disk {
        path: String,
        min_free_percent: f64,
    },

    /// Network connectivity check
    Network {
        target_host: String,
        timeout: Duration,
    },

    /// Custom health check
    Custom {
        check_name: String,
        implementation: String,
    },
}

/// Health check result
///
/// Result of executing a health check
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Check identifier
    pub check_id: String,

    /// Component identifier
    pub component_id: String,

    /// Check execution time
    pub timestamp: SystemTime,

    /// Check duration
    pub duration: Duration,

    /// Check status
    pub status: HealthCheckStatus,

    /// Response data
    pub response_data: Option<String>,

    /// Error information (if failed)
    pub error: Option<String>,

    /// Performance metrics
    pub metrics: HealthMetrics,

    /// Additional context
    pub context: HashMap<String, String>,
}

/// Health check status
#[derive(Debug, Clone, PartialEq)]
pub enum HealthCheckStatus {
    /// Check passed successfully
    Success,
    /// Check failed
    Failed,
    /// Check timed out
    Timeout,
    /// Check encountered an error
    Error,
    /// Check was skipped
    Skipped,
}

/// Health metrics captured during checks
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    /// Response time
    pub response_time: Duration,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Error rate
    pub error_rate: f64,
    /// Throughput
    pub throughput: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Component health status tracking
///
/// Tracks health status over time for trend analysis
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Current health state
    pub current_health: ComponentHealth,

    /// Health history
    pub health_history: VecDeque<HealthHistoryEntry>,

    /// Last successful check
    pub last_successful_check: Option<SystemTime>,

    /// Last failed check
    pub last_failed_check: Option<SystemTime>,

    /// Consecutive failures
    pub consecutive_failures: usize,

    /// Consecutive successes
    pub consecutive_successes: usize,

    /// Health trends
    pub trends: HealthTrends,

    /// Predictive health score
    pub predictive_score: Option<f64>,
}

/// Health history entry
#[derive(Debug, Clone)]
pub struct HealthHistoryEntry {
    /// Entry timestamp
    pub timestamp: SystemTime,
    /// Health state at this time
    pub health: ComponentHealth,
    /// Check result that led to this state
    pub check_result: Option<HealthCheckResult>,
    /// State transition reason
    pub transition_reason: String,
}

/// Health trends analysis
#[derive(Debug, Clone)]
pub struct HealthTrends {
    /// Performance trend direction
    pub performance_trend: TrendDirection,
    /// Error rate trend
    pub error_rate_trend: TrendDirection,
    /// Resource usage trend
    pub resource_trend: TrendDirection,
    /// Overall health trend
    pub overall_trend: TrendDirection,
    /// Trend confidence score
    pub confidence: f64,
    /// Prediction window
    pub prediction_window: Duration,
}

/// Trend direction enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Stable trend
    Stable,
    /// Degrading trend
    Degrading,
    /// Unknown/insufficient data
    Unknown,
}

/// Health checker trait
///
/// Interface for implementing different health checking strategies
pub trait HealthChecker: Send + Sync {
    /// Perform a health check
    fn check_health(&self, component_id: &str, config: &HealthCheckConfig) -> SklResult<HealthCheckResult>;

    /// Validate health check configuration
    fn validate_config(&self, config: &HealthCheckConfig) -> SklResult<()>;

    /// Get checker capabilities
    fn get_capabilities(&self) -> HealthCheckerCapabilities;
}

/// Health checker capabilities
#[derive(Debug, Clone)]
pub struct HealthCheckerCapabilities {
    /// Supported check types
    pub supported_types: Vec<String>,
    /// Minimum check interval
    pub min_interval: Duration,
    /// Maximum timeout
    pub max_timeout: Duration,
    /// Supports async checks
    pub async_support: bool,
    /// Supports batch checks
    pub batch_support: bool,
}

/// Component health tracker
///
/// Manages health monitoring for individual components
#[derive(Debug)]
pub struct ComponentHealthTracker {
    /// Component identifier
    component_id: String,

    /// Health configuration
    config: HealthCheckConfig,

    /// Current health status
    status: Arc<RwLock<HealthStatus>>,

    /// Health checker implementation
    checker: Arc<dyn HealthChecker>,

    /// Check scheduler
    scheduler: Arc<Mutex<HealthCheckScheduler>>,

    /// Alert manager
    alert_manager: Arc<Mutex<HealthAlertManager>>,

    /// Metrics collector
    metrics_collector: Arc<Mutex<HealthMetricsCollector>>,

    /// Tracker state
    state: Arc<RwLock<TrackerState>>,
}

/// Tracker state
#[derive(Debug, Clone, PartialEq)]
pub enum TrackerState {
    /// Tracker not started
    Stopped,
    /// Tracker running
    Running,
    /// Tracker paused
    Paused,
    /// Tracker failed
    Failed(String),
}

impl ComponentHealthTracker {
    /// Create a new health tracker
    pub fn new(
        component_id: String,
        config: HealthCheckConfig,
        checker: Arc<dyn HealthChecker>,
    ) -> Self {
        Self {
            component_id: component_id.clone(),
            config,
            status: Arc::new(RwLock::new(HealthStatus {
                current_health: ComponentHealth::Unknown {
                    last_check: SystemTime::now(),
                },
                health_history: VecDeque::new(),
                last_successful_check: None,
                last_failed_check: None,
                consecutive_failures: 0,
                consecutive_successes: 0,
                trends: HealthTrends {
                    performance_trend: TrendDirection::Unknown,
                    error_rate_trend: TrendDirection::Unknown,
                    resource_trend: TrendDirection::Unknown,
                    overall_trend: TrendDirection::Unknown,
                    confidence: 0.0,
                    prediction_window: Duration::from_hours(1),
                },
                predictive_score: None,
            })),
            checker,
            scheduler: Arc::new(Mutex::new(HealthCheckScheduler::new())),
            alert_manager: Arc::new(Mutex::new(HealthAlertManager::new())),
            metrics_collector: Arc::new(Mutex::new(HealthMetricsCollector::new())),
            state: Arc::new(RwLock::new(TrackerState::Stopped)),
        }
    }

    /// Start health monitoring
    pub fn start(&self) -> SklResult<()> {
        let mut state = self.state.write().unwrap();
        if *state != TrackerState::Stopped {
            return Err(SklearsError::InvalidState(
                "Tracker already running".to_string()
            ));
        }

        // Start scheduler
        {
            let mut scheduler = self.scheduler.lock().unwrap();
            scheduler.start(&self.component_id, &self.config)?;
        }

        *state = TrackerState::Running;
        Ok(())
    }

    /// Stop health monitoring
    pub fn stop(&self) -> SklResult<()> {
        let mut state = self.state.write().unwrap();
        if *state == TrackerState::Stopped {
            return Ok(());
        }

        // Stop scheduler
        {
            let mut scheduler = self.scheduler.lock().unwrap();
            scheduler.stop(&self.component_id)?;
        }

        *state = TrackerState::Stopped;
        Ok(())
    }

    /// Perform immediate health check
    pub fn check_now(&self) -> SklResult<HealthCheckResult> {
        let result = self.checker.check_health(&self.component_id, &self.config)?;
        self.process_check_result(&result)?;
        Ok(result)
    }

    /// Get current health status
    pub fn get_health(&self) -> ComponentHealth {
        self.status.read().unwrap().current_health.clone()
    }

    /// Get health status with history
    pub fn get_full_status(&self) -> HealthStatus {
        self.status.read().unwrap().clone()
    }

    /// Process health check result
    fn process_check_result(&self, result: &HealthCheckResult) -> SklResult<()> {
        let mut status = self.status.write().unwrap();

        // Update health state based on result
        let new_health = match result.status {
            HealthCheckStatus::Success => {
                status.consecutive_failures = 0;
                status.consecutive_successes += 1;
                status.last_successful_check = Some(result.timestamp);

                ComponentHealth::Healthy {
                    uptime: result.timestamp.duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or(Duration::ZERO),
                    performance_score: 1.0 - result.metrics.error_rate,
                }
            },
            HealthCheckStatus::Failed | HealthCheckStatus::Error => {
                status.consecutive_successes = 0;
                status.consecutive_failures += 1;
                status.last_failed_check = Some(result.timestamp);

                if status.consecutive_failures >= self.config.failure_threshold {
                    ComponentHealth::Failed {
                        failure_reason: result.error.clone()
                            .unwrap_or_else(|| "Health check failed".to_string()),
                        failure_time: result.timestamp,
                    }
                } else {
                    ComponentHealth::Unhealthy {
                        error_count: status.consecutive_failures,
                        last_error: result.error.clone()
                            .unwrap_or_else(|| "Unknown error".to_string()),
                    }
                }
            },
            HealthCheckStatus::Timeout => {
                status.consecutive_successes = 0;
                status.consecutive_failures += 1;
                status.last_failed_check = Some(result.timestamp);

                ComponentHealth::Degraded {
                    reason: "Health check timeout".to_string(),
                    impact_level: 0.5,
                }
            },
            HealthCheckStatus::Skipped => {
                // Don't change health state for skipped checks
                return Ok(());
            },
        };

        // Add to history
        status.health_history.push_back(HealthHistoryEntry {
            timestamp: result.timestamp,
            health: new_health.clone(),
            check_result: Some(result.clone()),
            transition_reason: format!("Health check result: {:?}", result.status),
        });

        // Limit history size
        if status.health_history.len() > 1000 {
            status.health_history.pop_front();
        }

        // Update current health
        status.current_health = new_health;

        // Update trends
        self.update_trends(&mut status)?;

        // Collect metrics
        {
            let mut collector = self.metrics_collector.lock().unwrap();
            collector.collect_metrics(&result)?;
        }

        // Check for alerts
        {
            let mut alert_manager = self.alert_manager.lock().unwrap();
            alert_manager.check_alerts(&status.current_health, &result)?;
        }

        Ok(())
    }

    /// Update health trends
    fn update_trends(&self, status: &mut HealthStatus) -> SklResult<()> {
        if status.health_history.len() < 10 {
            // Insufficient data for trend analysis
            return Ok(());
        }

        // Analyze trends based on recent history
        let recent_history: Vec<_> = status.health_history.iter()
            .rev()
            .take(50)
            .collect();

        // Performance trend analysis
        let performance_scores: Vec<f64> = recent_history.iter()
            .filter_map(|entry| {
                if let Some(ref result) = entry.check_result {
                    Some(1.0 - result.metrics.error_rate)
                } else {
                    None
                }
            })
            .collect();

        status.trends.performance_trend = self.analyze_trend(&performance_scores);

        // Error rate trend analysis
        let error_rates: Vec<f64> = recent_history.iter()
            .filter_map(|entry| {
                if let Some(ref result) = entry.check_result {
                    Some(result.metrics.error_rate)
                } else {
                    None
                }
            })
            .collect();

        status.trends.error_rate_trend = self.analyze_trend(&error_rates);

        // Resource usage trend analysis
        let resource_usage: Vec<f64> = recent_history.iter()
            .filter_map(|entry| {
                if let Some(ref result) = entry.check_result {
                    Some(result.metrics.resource_utilization)
                } else {
                    None
                }
            })
            .collect();

        status.trends.resource_trend = self.analyze_trend(&resource_usage);

        // Overall trend
        status.trends.overall_trend = self.calculate_overall_trend(&status.trends);

        // Calculate confidence
        status.trends.confidence = self.calculate_trend_confidence(&recent_history);

        Ok(())
    }

    /// Analyze trend direction from data points
    fn analyze_trend(&self, data: &[f64]) -> TrendDirection {
        if data.len() < 5 {
            return TrendDirection::Unknown;
        }

        // Simple linear regression to determine trend
        let n = data.len() as f64;
        let sum_x: f64 = (0..data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = data.iter().sum();
        let sum_xy: f64 = data.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

        if slope > 0.01 {
            TrendDirection::Improving
        } else if slope < -0.01 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate overall trend from individual trends
    fn calculate_overall_trend(&self, trends: &HealthTrends) -> TrendDirection {
        let improving_count = [
            &trends.performance_trend,
            &trends.error_rate_trend,
            &trends.resource_trend,
        ]
        .iter()
        .filter(|&&trend| trend == TrendDirection::Improving)
        .count();

        let degrading_count = [
            &trends.performance_trend,
            &trends.error_rate_trend,
            &trends.resource_trend,
        ]
        .iter()
        .filter(|&&trend| trend == TrendDirection::Degrading)
        .count();

        if improving_count > degrading_count {
            TrendDirection::Improving
        } else if degrading_count > improving_count {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate trend confidence score
    fn calculate_trend_confidence(&self, history: &[&HealthHistoryEntry]) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        // Confidence based on data quality and consistency
        let data_quality = history.len() as f64 / 50.0; // Normalize to 50 data points
        let consistency = self.calculate_consistency(history);

        (data_quality * consistency).min(1.0)
    }

    /// Calculate data consistency
    fn calculate_consistency(&self, _history: &[&HealthHistoryEntry]) -> f64 {
        // Simplified consistency calculation
        // In practice, this would analyze variance and stability
        0.8
    }
}

/// Health check scheduler
///
/// Manages scheduling of health checks for components
#[derive(Debug)]
pub struct HealthCheckScheduler {
    /// Scheduled components
    scheduled_components: HashMap<String, ScheduledCheck>,
    /// Scheduler state
    running: bool,
}

/// Scheduled check information
#[derive(Debug, Clone)]
pub struct ScheduledCheck {
    /// Component identifier
    pub component_id: String,
    /// Check configuration
    pub config: HealthCheckConfig,
    /// Next check time
    pub next_check: SystemTime,
    /// Last check time
    pub last_check: Option<SystemTime>,
}

impl HealthCheckScheduler {
    /// Create a new scheduler
    pub fn new() -> Self {
        Self {
            scheduled_components: HashMap::new(),
            running: false,
        }
    }

    /// Start the scheduler
    pub fn start(&mut self, component_id: &str, config: &HealthCheckConfig) -> SklResult<()> {
        self.scheduled_components.insert(
            component_id.to_string(),
            ScheduledCheck {
                component_id: component_id.to_string(),
                config: config.clone(),
                next_check: SystemTime::now() + config.interval,
                last_check: None,
            },
        );

        self.running = true;
        Ok(())
    }

    /// Stop the scheduler
    pub fn stop(&mut self, component_id: &str) -> SklResult<()> {
        self.scheduled_components.remove(component_id);

        if self.scheduled_components.is_empty() {
            self.running = false;
        }

        Ok(())
    }

    /// Get next scheduled check
    pub fn get_next_check(&self) -> Option<&ScheduledCheck> {
        self.scheduled_components
            .values()
            .min_by_key(|check| check.next_check)
    }
}

/// Health alert manager
///
/// Manages health-related alerts and notifications
#[derive(Debug)]
pub struct HealthAlertManager {
    /// Alert rules
    alert_rules: Vec<HealthAlertRule>,
    /// Alert state
    alert_state: HashMap<String, AlertState>,
}

/// Health alert rule
#[derive(Debug, Clone)]
pub struct HealthAlertRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule condition
    pub condition: HealthAlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Rule enabled
    pub enabled: bool,
}

/// Health alert condition
#[derive(Debug, Clone)]
pub enum HealthAlertCondition {
    /// Health state matches
    HealthState(ComponentHealth),
    /// Consecutive failures
    ConsecutiveFailures(usize),
    /// Error rate threshold
    ErrorRate(f64),
    /// Performance degradation
    PerformanceDegradation(f64),
}

/// Alert severity
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert
    Critical,
}

/// Alert state
#[derive(Debug, Clone)]
pub struct AlertState {
    /// Alert active
    pub active: bool,
    /// First triggered
    pub first_triggered: Option<SystemTime>,
    /// Last triggered
    pub last_triggered: Option<SystemTime>,
    /// Trigger count
    pub trigger_count: usize,
}

impl HealthAlertManager {
    /// Create a new alert manager
    pub fn new() -> Self {
        Self {
            alert_rules: Vec::new(),
            alert_state: HashMap::new(),
        }
    }

    /// Add alert rule
    pub fn add_rule(&mut self, rule: HealthAlertRule) {
        self.alert_rules.push(rule);
    }

    /// Check for alerts
    pub fn check_alerts(
        &mut self,
        health: &ComponentHealth,
        result: &HealthCheckResult,
    ) -> SklResult<()> {
        for rule in &self.alert_rules {
            if !rule.enabled {
                continue;
            }

            let should_trigger = self.evaluate_condition(&rule.condition, health, result);

            if should_trigger {
                self.trigger_alert(&rule.rule_id, &rule.severity)?;
            }
        }

        Ok(())
    }

    /// Evaluate alert condition
    fn evaluate_condition(
        &self,
        condition: &HealthAlertCondition,
        health: &ComponentHealth,
        result: &HealthCheckResult,
    ) -> bool {
        match condition {
            HealthAlertCondition::HealthState(expected) => {
                std::mem::discriminant(health) == std::mem::discriminant(expected)
            },
            HealthAlertCondition::ConsecutiveFailures(_threshold) => {
                matches!(result.status, HealthCheckStatus::Failed | HealthCheckStatus::Error)
            },
            HealthAlertCondition::ErrorRate(threshold) => {
                result.metrics.error_rate > *threshold
            },
            HealthAlertCondition::PerformanceDegradation(threshold) => {
                result.metrics.resource_utilization > *threshold
            },
        }
    }

    /// Trigger alert
    fn trigger_alert(&mut self, rule_id: &str, _severity: &AlertSeverity) -> SklResult<()> {
        let now = SystemTime::now();

        let state = self.alert_state.entry(rule_id.to_string())
            .or_insert_with(|| AlertState {
                active: false,
                first_triggered: None,
                last_triggered: None,
                trigger_count: 0,
            });

        if !state.active {
            state.first_triggered = Some(now);
            state.active = true;
        }

        state.last_triggered = Some(now);
        state.trigger_count += 1;

        Ok(())
    }
}

/// Health metrics collector
///
/// Collects and aggregates health metrics
#[derive(Debug)]
pub struct HealthMetricsCollector {
    /// Collected metrics
    metrics: VecDeque<HealthCheckResult>,
    /// Aggregated statistics
    stats: HealthStatistics,
}

/// Health statistics
#[derive(Debug, Clone)]
pub struct HealthStatistics {
    /// Total checks performed
    pub total_checks: u64,
    /// Successful checks
    pub successful_checks: u64,
    /// Failed checks
    pub failed_checks: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Average error rate
    pub avg_error_rate: f64,
    /// Uptime percentage
    pub uptime_percentage: f64,
}

impl HealthMetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: VecDeque::new(),
            stats: HealthStatistics {
                total_checks: 0,
                successful_checks: 0,
                failed_checks: 0,
                avg_response_time: Duration::ZERO,
                avg_error_rate: 0.0,
                uptime_percentage: 100.0,
            },
        }
    }

    /// Collect metrics from health check result
    pub fn collect_metrics(&mut self, result: &HealthCheckResult) -> SklResult<()> {
        self.metrics.push_back(result.clone());

        // Limit metrics storage
        if self.metrics.len() > 10000 {
            self.metrics.pop_front();
        }

        // Update statistics
        self.update_statistics();

        Ok(())
    }

    /// Update aggregated statistics
    fn update_statistics(&mut self) {
        if self.metrics.is_empty() {
            return;
        }

        self.stats.total_checks = self.metrics.len() as u64;
        self.stats.successful_checks = self.metrics.iter()
            .filter(|m| m.status == HealthCheckStatus::Success)
            .count() as u64;
        self.stats.failed_checks = self.stats.total_checks - self.stats.successful_checks;

        self.stats.avg_response_time = Duration::from_nanos(
            self.metrics.iter()
                .map(|m| m.duration.as_nanos())
                .sum::<u128>() / self.metrics.len() as u128
        );

        self.stats.avg_error_rate = self.metrics.iter()
            .map(|m| m.metrics.error_rate)
            .sum::<f64>() / self.metrics.len() as f64;

        self.stats.uptime_percentage = (self.stats.successful_checks as f64 / self.stats.total_checks as f64) * 100.0;
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> HealthStatistics {
        self.stats.clone()
    }
}

/// Component health monitor
///
/// Coordinates health monitoring across multiple components
#[derive(Debug)]
pub struct ComponentHealthMonitor {
    /// Component trackers
    trackers: Arc<RwLock<HashMap<String, ComponentHealthTracker>>>,
    /// Monitor configuration
    config: HealthMonitorConfig,
    /// Monitor state
    state: Arc<RwLock<MonitorState>>,
}

/// Health monitor configuration
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Enable predictive analysis
    pub enable_predictive: bool,
    /// Enable trend analysis
    pub enable_trends: bool,
    /// Enable alerts
    pub enable_alerts: bool,
    /// Global check interval
    pub global_interval: Duration,
    /// Maximum concurrent checks
    pub max_concurrent_checks: usize,
}

/// Monitor state
#[derive(Debug, Clone, PartialEq)]
pub enum MonitorState {
    /// Monitor stopped
    Stopped,
    /// Monitor starting
    Starting,
    /// Monitor running
    Running,
    /// Monitor stopping
    Stopping,
    /// Monitor failed
    Failed(String),
}

impl ComponentHealthMonitor {
    /// Create a new health monitor
    pub fn new() -> Self {
        Self {
            trackers: Arc::new(RwLock::new(HashMap::new())),
            config: HealthMonitorConfig {
                enable_predictive: true,
                enable_trends: true,
                enable_alerts: true,
                global_interval: Duration::from_secs(30),
                max_concurrent_checks: 10,
            },
            state: Arc::new(RwLock::new(MonitorState::Stopped)),
        }
    }

    /// Initialize the monitor
    pub fn initialize(&self) -> SklResult<()> {
        let mut state = self.state.write().unwrap();
        *state = MonitorState::Starting;
        *state = MonitorState::Running;
        Ok(())
    }

    /// Shutdown the monitor
    pub fn shutdown(&self) -> SklResult<()> {
        let mut state = self.state.write().unwrap();
        *state = MonitorState::Stopping;

        // Stop all trackers
        let trackers = self.trackers.read().unwrap();
        for tracker in trackers.values() {
            tracker.stop()?;
        }

        *state = MonitorState::Stopped;
        Ok(())
    }

    /// Add component for monitoring
    pub fn add_component(
        &self,
        component_id: String,
        config: HealthCheckConfig,
        checker: Arc<dyn HealthChecker>,
    ) -> SklResult<()> {
        let tracker = ComponentHealthTracker::new(component_id.clone(), config, checker);

        // Start the tracker if monitor is running
        if *self.state.read().unwrap() == MonitorState::Running {
            tracker.start()?;
        }

        let mut trackers = self.trackers.write().unwrap();
        trackers.insert(component_id, tracker);

        Ok(())
    }

    /// Remove component from monitoring
    pub fn remove_component(&self, component_id: &str) -> SklResult<()> {
        let mut trackers = self.trackers.write().unwrap();
        if let Some(tracker) = trackers.remove(component_id) {
            tracker.stop()?;
        }
        Ok(())
    }

    /// Get component health
    pub fn get_component_health(&self, component_id: &str) -> SklResult<ComponentHealth> {
        let trackers = self.trackers.read().unwrap();
        trackers.get(component_id)
            .map(|tracker| tracker.get_health())
            .ok_or_else(|| SklearsError::Other("Component not found".to_string()))
    }

    /// Get all component health statuses
    pub fn get_all_health(&self) -> HashMap<String, ComponentHealth> {
        let trackers = self.trackers.read().unwrap();
        trackers.iter()
            .map(|(id, tracker)| (id.clone(), tracker.get_health()))
            .collect()
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            endpoint: "health".to_string(),
            expected_response: None,
            failure_threshold: 3,
            success_threshold: 2,
            check_type: HealthCheckType::Custom {
                check_name: "default".to_string(),
                implementation: "noop".to_string(),
            },
            predictive_analysis: false,
            metadata: HashMap::new(),
        }
    }
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            response_time: Duration::ZERO,
            resource_utilization: 0.0,
            error_rate: 0.0,
            throughput: 0.0,
            custom_metrics: HashMap::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_creation() {
        let health = ComponentHealth::Healthy {
            uptime: Duration::from_secs(3600),
            performance_score: 0.95,
        };

        match health {
            ComponentHealth::Healthy { uptime, performance_score } => {
                assert_eq!(uptime, Duration::from_secs(3600));
                assert_eq!(performance_score, 0.95);
            },
            _ => panic!("Expected healthy status"),
        }
    }

    #[test]
    fn test_health_check_config() {
        let config = HealthCheckConfig::default();
        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.success_threshold, 2);
    }

    #[test]
    fn test_trend_direction() {
        let trend = TrendDirection::Improving;
        assert_eq!(trend, TrendDirection::Improving);
        assert_ne!(trend, TrendDirection::Degrading);
    }
}