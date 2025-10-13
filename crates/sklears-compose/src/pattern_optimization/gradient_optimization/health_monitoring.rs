//! Health Monitoring Module for Gradient Optimization
//!
//! This module provides comprehensive health monitoring, trend analysis, anomaly detection,
//! and alert management for all components in the gradient optimization system.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant, SystemTime};
use std::fmt;
use scirs2_core::error::{CoreError, Result as SklResult};
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Random, rng};
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use serde::{Deserialize, Serialize};
use tokio::{sync::{broadcast, mpsc, oneshot}, time::{interval, sleep, timeout}};

/// Health monitoring system for comprehensive system health tracking
#[derive(Debug)]
pub struct HealthMonitor {
    pub health_checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
    pub scheduler: Arc<HealthCheckScheduler>,
    pub result_aggregator: Arc<HealthCheckAggregator>,
    pub dependency_tracker: Arc<HealthDependencyTracker>,
    pub alert_integration: Arc<HealthAlertIntegration>,
    pub trend_analyzer: Arc<HealthTrendAnalyzer>,
    pub anomaly_detector: Arc<HealthAnomalyDetector>,
    pub alert_manager: Arc<HealthAlertManager>,
    pub auto_recovery: Arc<AutoRecoveryManager>,
    pub metrics_collector: Arc<HealthMetricsCollector>,
    pub configuration: HealthMonitorConfig,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(config: HealthMonitorConfig) -> Self {
        Self {
            health_checks: Arc::new(RwLock::new(HashMap::new())),
            scheduler: Arc::new(HealthCheckScheduler::new()),
            result_aggregator: Arc::new(HealthCheckAggregator::new()),
            dependency_tracker: Arc::new(HealthDependencyTracker::new()),
            alert_integration: Arc::new(HealthAlertIntegration::new()),
            trend_analyzer: Arc::new(HealthTrendAnalyzer::new()),
            anomaly_detector: Arc::new(HealthAnomalyDetector::new()),
            alert_manager: Arc::new(HealthAlertManager::new()),
            auto_recovery: Arc::new(AutoRecoveryManager::new()),
            metrics_collector: Arc::new(HealthMetricsCollector::new()),
            configuration: config,
        }
    }

    /// Start health monitoring
    pub async fn start(&self) -> SklResult<()> {
        // Start scheduler
        self.scheduler.start().await?;

        // Start trend analyzer
        self.trend_analyzer.start().await?;

        // Start anomaly detector
        self.anomaly_detector.start().await?;

        // Start alert manager
        self.alert_manager.start().await?;

        // Start auto recovery
        self.auto_recovery.start().await?;

        Ok(())
    }

    /// Stop health monitoring
    pub async fn stop(&self) -> SklResult<()> {
        // Stop components in reverse order
        self.auto_recovery.stop().await?;
        self.alert_manager.stop().await?;
        self.anomaly_detector.stop().await?;
        self.trend_analyzer.stop().await?;
        self.scheduler.stop().await?;

        Ok(())
    }

    /// Register a new health check
    pub async fn register_health_check(&self, health_check: HealthCheck) -> SklResult<()> {
        let mut checks = self.health_checks.write().unwrap();
        checks.insert(health_check.name.clone(), health_check.clone());

        // Schedule the health check
        self.scheduler.schedule_check(health_check).await?;

        Ok(())
    }

    /// Unregister a health check
    pub async fn unregister_health_check(&self, name: &str) -> SklResult<()> {
        let mut checks = self.health_checks.write().unwrap();
        checks.remove(name);

        // Unschedule the health check
        self.scheduler.unschedule_check(name).await?;

        Ok(())
    }

    /// Perform immediate health check
    pub async fn perform_health_check(&self, name: &str) -> SklResult<HealthCheckResult> {
        let checks = self.health_checks.read().unwrap();
        let health_check = checks.get(name)
            .ok_or_else(|| CoreError::InvalidOperation(format!("Health check '{}' not found", name)))?;

        let result = self.execute_health_check(health_check).await?;

        // Record result
        self.result_aggregator.record_result(result.clone()).await?;

        // Check for alerts
        self.alert_manager.check_result(&result).await?;

        Ok(result)
    }

    /// Get health status summary
    pub async fn get_health_summary(&self) -> SklResult<HealthSummary> {
        self.result_aggregator.get_summary().await
    }

    /// Get detailed health report
    pub async fn get_health_report(&self) -> SklResult<HealthReport> {
        let summary = self.get_health_summary().await?;
        let trends = self.trend_analyzer.get_current_trends().await?;
        let anomalies = self.anomaly_detector.get_recent_anomalies().await?;
        let alerts = self.alert_manager.get_active_alerts().await?;

        Ok(HealthReport {
            summary,
            trends,
            anomalies,
            alerts,
            timestamp: Instant::now(),
        })
    }

    /// Execute a single health check
    async fn execute_health_check(&self, health_check: &HealthCheck) -> SklResult<HealthCheckResult> {
        let start_time = Instant::now();

        let result = timeout(
            health_check.timeout,
            health_check.executor.execute()
        ).await;

        let execution_time = start_time.elapsed();

        match result {
            Ok(Ok(mut result)) => {
                result.response_time = execution_time;
                result.timestamp = start_time;
                Ok(result)
            }
            Ok(Err(e)) => Ok(HealthCheckResult {
                check_name: health_check.name.clone(),
                check_type: health_check.check_type.clone(),
                healthy: false,
                message: format!("Health check failed: {}", e),
                details: HashMap::new(),
                response_time: execution_time,
                timestamp: start_time,
                warnings: vec![],
                metrics: HashMap::new(),
            }),
            Err(_) => Ok(HealthCheckResult {
                check_name: health_check.name.clone(),
                check_type: health_check.check_type.clone(),
                healthy: false,
                message: "Health check timed out".to_string(),
                details: HashMap::new(),
                response_time: execution_time,
                timestamp: start_time,
                warnings: vec!["Timeout occurred".to_string()],
                metrics: HashMap::new(),
            }),
        }
    }
}

/// Configuration for health monitoring
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    pub default_check_interval: Duration,
    pub default_check_timeout: Duration,
    pub trend_analysis_window: Duration,
    pub anomaly_detection_sensitivity: f64,
    pub alert_cooldown_period: Duration,
    pub auto_recovery_enabled: bool,
    pub max_result_history: usize,
    pub enable_detailed_logging: bool,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            default_check_interval: Duration::from_seconds(30),
            default_check_timeout: Duration::from_seconds(10),
            trend_analysis_window: Duration::from_minutes(15),
            anomaly_detection_sensitivity: 0.05,
            alert_cooldown_period: Duration::from_minutes(5),
            auto_recovery_enabled: true,
            max_result_history: 1000,
            enable_detailed_logging: false,
        }
    }
}

/// Individual health check definition
#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub check_type: HealthCheckType,
    pub executor: Arc<dyn HealthCheckExecutor>,
    pub schedule: HealthCheckSchedule,
    pub timeout: Duration,
    pub dependencies: Vec<String>,
    pub is_critical: bool,
    pub tags: HashMap<String, String>,
    pub enabled: bool,
}

impl HealthCheck {
    /// Create a new health check
    pub fn new(
        name: String,
        check_type: HealthCheckType,
        executor: Arc<dyn HealthCheckExecutor>,
    ) -> Self {
        Self {
            name,
            check_type,
            executor,
            schedule: HealthCheckSchedule::default(),
            timeout: Duration::from_seconds(10),
            dependencies: vec![],
            is_critical: false,
            tags: HashMap::new(),
            enabled: true,
        }
    }

    /// Set schedule for the health check
    pub fn with_schedule(mut self, schedule: HealthCheckSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Set timeout for the health check
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Mark as critical health check
    pub fn critical(mut self) -> Self {
        self.is_critical = true;
        self
    }

    /// Add dependencies
    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }

    /// Add tag
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }
}

/// Health check types
#[derive(Debug, Clone, PartialEq)]
pub enum HealthCheckType {
    Liveness,
    Readiness,
    Performance,
    Dependency,
    Resource,
    Custom(String),
}

/// Health check executor trait
pub trait HealthCheckExecutor: Send + Sync + std::fmt::Debug {
    fn execute(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = SklResult<HealthCheckResult>> + Send>>;
    fn get_check_type(&self) -> HealthCheckType;
    fn get_description(&self) -> String;
}

/// Health check schedule
#[derive(Debug, Clone)]
pub struct HealthCheckSchedule {
    pub interval: Duration,
    pub initial_delay: Duration,
    pub jitter: Duration,
    pub enabled: bool,
    pub cron_expression: Option<String>,
}

impl Default for HealthCheckSchedule {
    fn default() -> Self {
        Self {
            interval: Duration::from_seconds(30),
            initial_delay: Duration::from_seconds(0),
            jitter: Duration::from_seconds(5),
            enabled: true,
            cron_expression: None,
        }
    }
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub check_name: String,
    pub check_type: HealthCheckType,
    pub healthy: bool,
    pub message: String,
    pub details: HashMap<String, String>,
    pub response_time: Duration,
    pub timestamp: Instant,
    pub warnings: Vec<String>,
    pub metrics: HashMap<String, f64>,
}

impl HealthCheckResult {
    /// Create a healthy result
    pub fn healthy(check_name: String, check_type: HealthCheckType) -> Self {
        Self {
            check_name,
            check_type,
            healthy: true,
            message: "Check passed".to_string(),
            details: HashMap::new(),
            response_time: Duration::from_millis(0),
            timestamp: Instant::now(),
            warnings: vec![],
            metrics: HashMap::new(),
        }
    }

    /// Create an unhealthy result
    pub fn unhealthy(check_name: String, check_type: HealthCheckType, message: String) -> Self {
        Self {
            check_name,
            check_type,
            healthy: false,
            message,
            details: HashMap::new(),
            response_time: Duration::from_millis(0),
            timestamp: Instant::now(),
            warnings: vec![],
            metrics: HashMap::new(),
        }
    }

    /// Add detail to the result
    pub fn with_detail(mut self, key: String, value: String) -> Self {
        self.details.insert(key, value);
        self
    }

    /// Add warning to the result
    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Add metric to the result
    pub fn with_metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }
}

/// Health check scheduler
#[derive(Debug)]
pub struct HealthCheckScheduler {
    pub scheduled_checks: Arc<RwLock<HashMap<String, ScheduledCheck>>>,
    pub is_running: AtomicBool,
    pub scheduler_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl HealthCheckScheduler {
    pub fn new() -> Self {
        Self {
            scheduled_checks: Arc::new(RwLock::new(HashMap::new())),
            is_running: AtomicBool::new(false),
            scheduler_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) -> SklResult<()> {
        self.is_running.store(true, Ordering::SeqCst);

        let scheduled_checks = self.scheduled_checks.clone();
        let is_running = self.is_running.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_seconds(1));

            while is_running.load(Ordering::SeqCst) {
                interval.tick().await;

                let checks = scheduled_checks.read().unwrap();
                let current_time = Instant::now();

                for (name, scheduled_check) in checks.iter() {
                    if scheduled_check.is_due(current_time) {
                        // Execute check asynchronously
                        let check = scheduled_check.health_check.clone();
                        tokio::spawn(async move {
                            // Execute the health check
                            // In a real implementation, this would be connected to the health monitor
                        });
                    }
                }
            }
        });

        *self.scheduler_handle.lock().unwrap() = Some(handle);

        Ok(())
    }

    pub async fn stop(&self) -> SklResult<()> {
        self.is_running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.scheduler_handle.lock().unwrap().take() {
            handle.abort();
        }

        Ok(())
    }

    pub async fn schedule_check(&self, health_check: HealthCheck) -> SklResult<()> {
        let scheduled_check = ScheduledCheck::new(health_check);
        let mut checks = self.scheduled_checks.write().unwrap();
        checks.insert(scheduled_check.health_check.name.clone(), scheduled_check);
        Ok(())
    }

    pub async fn unschedule_check(&self, name: &str) -> SklResult<()> {
        let mut checks = self.scheduled_checks.write().unwrap();
        checks.remove(name);
        Ok(())
    }
}

/// Scheduled health check
#[derive(Debug)]
pub struct ScheduledCheck {
    pub health_check: HealthCheck,
    pub next_execution: Instant,
    pub last_execution: Option<Instant>,
    pub execution_count: usize,
}

impl ScheduledCheck {
    pub fn new(health_check: HealthCheck) -> Self {
        let next_execution = Instant::now() + health_check.schedule.initial_delay;

        Self {
            health_check,
            next_execution,
            last_execution: None,
            execution_count: 0,
        }
    }

    pub fn is_due(&self, current_time: Instant) -> bool {
        current_time >= self.next_execution
    }

    pub fn update_next_execution(&mut self) {
        self.last_execution = Some(Instant::now());
        self.execution_count += 1;

        // Add jitter to prevent thundering herd
        let jitter = Duration::from_millis(
            (rng().random::<f64>() * self.health_check.schedule.jitter.as_millis() as f64) as u64
        );

        self.next_execution = Instant::now() + self.health_check.schedule.interval + jitter;
    }
}

/// Health check result aggregator
#[derive(Debug)]
pub struct HealthCheckAggregator {
    pub results: Arc<RwLock<HashMap<String, VecDeque<HealthCheckResult>>>>,
    pub max_results_per_check: usize,
}

impl HealthCheckAggregator {
    pub fn new() -> Self {
        Self {
            results: Arc::new(RwLock::new(HashMap::new())),
            max_results_per_check: 100,
        }
    }

    pub async fn record_result(&self, result: HealthCheckResult) -> SklResult<()> {
        let mut results = self.results.write().unwrap();
        let check_results = results.entry(result.check_name.clone()).or_insert_with(VecDeque::new);

        check_results.push_back(result);

        if check_results.len() > self.max_results_per_check {
            check_results.pop_front();
        }

        Ok(())
    }

    pub async fn get_summary(&self) -> SklResult<HealthSummary> {
        let results = self.results.read().unwrap();
        let mut total_checks = 0;
        let mut healthy_checks = 0;
        let mut check_statuses = HashMap::new();

        for (check_name, check_results) in results.iter() {
            if let Some(latest_result) = check_results.back() {
                total_checks += 1;
                if latest_result.healthy {
                    healthy_checks += 1;
                }

                check_statuses.insert(check_name.clone(), CheckStatus {
                    healthy: latest_result.healthy,
                    last_check: latest_result.timestamp,
                    response_time: latest_result.response_time,
                    message: latest_result.message.clone(),
                    check_type: latest_result.check_type.clone(),
                });
            }
        }

        let overall_health = if total_checks == 0 {
            true
        } else {
            healthy_checks == total_checks
        };

        let health_score = if total_checks == 0 {
            1.0
        } else {
            healthy_checks as f64 / total_checks as f64
        };

        Ok(HealthSummary {
            overall_health,
            health_score,
            total_checks,
            healthy_checks,
            unhealthy_checks: total_checks - healthy_checks,
            check_statuses,
            last_updated: Instant::now(),
        })
    }

    pub async fn get_check_history(&self, check_name: &str, limit: usize) -> SklResult<Vec<HealthCheckResult>> {
        let results = self.results.read().unwrap();

        if let Some(check_results) = results.get(check_name) {
            let history: Vec<HealthCheckResult> = check_results
                .iter()
                .rev()
                .take(limit)
                .cloned()
                .collect();
            Ok(history)
        } else {
            Ok(vec![])
        }
    }
}

/// Health summary
#[derive(Debug, Clone)]
pub struct HealthSummary {
    pub overall_health: bool,
    pub health_score: f64,
    pub total_checks: usize,
    pub healthy_checks: usize,
    pub unhealthy_checks: usize,
    pub check_statuses: HashMap<String, CheckStatus>,
    pub last_updated: Instant,
}

/// Individual check status
#[derive(Debug, Clone)]
pub struct CheckStatus {
    pub healthy: bool,
    pub last_check: Instant,
    pub response_time: Duration,
    pub message: String,
    pub check_type: HealthCheckType,
}

/// Health dependency tracker
#[derive(Debug)]
pub struct HealthDependencyTracker {
    pub dependencies: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl HealthDependencyTracker {
    pub fn new() -> Self {
        Self {
            dependencies: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn add_dependency(&self, check_name: String, dependencies: Vec<String>) {
        let mut deps = self.dependencies.write().unwrap();
        deps.insert(check_name, dependencies);
    }

    pub fn get_dependencies(&self, check_name: &str) -> Vec<String> {
        let deps = self.dependencies.read().unwrap();
        deps.get(check_name).cloned().unwrap_or_default()
    }

    pub fn check_dependencies_healthy(&self, check_name: &str, health_summary: &HealthSummary) -> bool {
        let dependencies = self.get_dependencies(check_name);

        dependencies.iter().all(|dep| {
            health_summary.check_statuses
                .get(dep)
                .map(|status| status.healthy)
                .unwrap_or(false)
        })
    }
}

/// Health alert integration
#[derive(Debug)]
pub struct HealthAlertIntegration {
    pub alert_channels: Arc<RwLock<Vec<AlertChannel>>>,
}

impl HealthAlertIntegration {
    pub fn new() -> Self {
        Self {
            alert_channels: Arc::new(RwLock::new(vec![])),
        }
    }

    pub fn add_alert_channel(&self, channel: AlertChannel) {
        let mut channels = self.alert_channels.write().unwrap();
        channels.push(channel);
    }

    pub async fn send_alert(&self, alert: HealthAlert) -> SklResult<()> {
        let channels = self.alert_channels.read().unwrap();

        for channel in channels.iter() {
            channel.send_alert(&alert).await?;
        }

        Ok(())
    }
}

/// Alert channel trait
pub trait AlertChannel: Send + Sync + std::fmt::Debug {
    fn send_alert(&self, alert: &HealthAlert) -> std::pin::Pin<Box<dyn std::future::Future<Output = SklResult<()>> + Send>>;
    fn get_channel_type(&self) -> String;
    fn is_enabled(&self) -> bool;
}

/// Health alert
#[derive(Debug, Clone)]
pub struct HealthAlert {
    pub alert_id: String,
    pub check_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub details: HashMap<String, String>,
    pub timestamp: Instant,
    pub resolved: bool,
}

/// Alert severity
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Health trend analyzer
#[derive(Debug)]
pub struct HealthTrendAnalyzer {
    pub trend_data: Arc<RwLock<HashMap<String, TrendData>>>,
    pub analysis_window: Duration,
    pub is_running: AtomicBool,
    pub analyzer_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl HealthTrendAnalyzer {
    pub fn new() -> Self {
        Self {
            trend_data: Arc::new(RwLock::new(HashMap::new())),
            analysis_window: Duration::from_minutes(15),
            is_running: AtomicBool::new(false),
            analyzer_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) -> SklResult<()> {
        self.is_running.store(true, Ordering::SeqCst);

        let trend_data = self.trend_data.clone();
        let is_running = self.is_running.clone();
        let analysis_window = self.analysis_window;

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_minutes(1));

            while is_running.load(Ordering::SeqCst) {
                interval.tick().await;

                // Perform trend analysis
                let mut data = trend_data.write().unwrap();
                let current_time = Instant::now();

                for trend in data.values_mut() {
                    trend.update_trends(current_time, analysis_window);
                }
            }
        });

        *self.analyzer_handle.lock().unwrap() = Some(handle);

        Ok(())
    }

    pub async fn stop(&self) -> SklResult<()> {
        self.is_running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.analyzer_handle.lock().unwrap().take() {
            handle.abort();
        }

        Ok(())
    }

    pub fn record_data_point(&self, check_name: String, value: f64, timestamp: Instant) {
        let mut data = self.trend_data.write().unwrap();
        let trend = data.entry(check_name).or_insert_with(TrendData::new);
        trend.add_data_point(value, timestamp);
    }

    pub async fn get_current_trends(&self) -> SklResult<HashMap<String, TrendInfo>> {
        let data = self.trend_data.read().unwrap();
        let mut trends = HashMap::new();

        for (check_name, trend_data) in data.iter() {
            trends.insert(check_name.clone(), trend_data.get_trend_info());
        }

        Ok(trends)
    }
}

/// Trend data for a specific metric
#[derive(Debug)]
pub struct TrendData {
    pub data_points: VecDeque<DataPoint>,
    pub current_trend: TrendDirection,
    pub trend_strength: f64,
    pub last_analysis: Option<Instant>,
}

impl TrendData {
    pub fn new() -> Self {
        Self {
            data_points: VecDeque::new(),
            current_trend: TrendDirection::Stable,
            trend_strength: 0.0,
            last_analysis: None,
        }
    }

    pub fn add_data_point(&mut self, value: f64, timestamp: Instant) {
        self.data_points.push_back(DataPoint { value, timestamp });

        // Keep only recent data points
        while self.data_points.len() > 1000 {
            self.data_points.pop_front();
        }
    }

    pub fn update_trends(&mut self, current_time: Instant, window: Duration) {
        let cutoff_time = current_time - window;

        // Remove old data points
        while let Some(front) = self.data_points.front() {
            if front.timestamp < cutoff_time {
                self.data_points.pop_front();
            } else {
                break;
            }
        }

        // Calculate trend
        if self.data_points.len() >= 2 {
            let (trend, strength) = self.calculate_trend();
            self.current_trend = trend;
            self.trend_strength = strength;
        }

        self.last_analysis = Some(current_time);
    }

    pub fn get_trend_info(&self) -> TrendInfo {
        TrendInfo {
            direction: self.current_trend.clone(),
            strength: self.trend_strength,
            data_points: self.data_points.len(),
            last_analysis: self.last_analysis,
        }
    }

    fn calculate_trend(&self) -> (TrendDirection, f64) {
        if self.data_points.len() < 2 {
            return (TrendDirection::Stable, 0.0);
        }

        // Simple linear regression for trend calculation
        let n = self.data_points.len() as f64;
        let sum_x: f64 = (0..self.data_points.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.data_points.iter().map(|p| p.value).sum();
        let sum_xy: f64 = self.data_points.iter().enumerate()
            .map(|(i, p)| i as f64 * p.value).sum();
        let sum_x_squared: f64 = (0..self.data_points.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x.powi(2));

        let direction = if slope > 0.1 {
            TrendDirection::Increasing
        } else if slope < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let strength = slope.abs().min(1.0);

        (direction, strength)
    }
}

/// Data point for trend analysis
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub value: f64,
    pub timestamp: Instant,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Trend information
#[derive(Debug, Clone)]
pub struct TrendInfo {
    pub direction: TrendDirection,
    pub strength: f64,
    pub data_points: usize,
    pub last_analysis: Option<Instant>,
}

/// Health anomaly detector
#[derive(Debug)]
pub struct HealthAnomalyDetector {
    pub baseline_data: Arc<RwLock<HashMap<String, BaselineData>>>,
    pub anomalies: Arc<RwLock<VecDeque<Anomaly>>>,
    pub sensitivity: f64,
    pub is_running: AtomicBool,
    pub detector_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl HealthAnomalyDetector {
    pub fn new() -> Self {
        Self {
            baseline_data: Arc::new(RwLock::new(HashMap::new())),
            anomalies: Arc::new(RwLock::new(VecDeque::new())),
            sensitivity: 0.05,
            is_running: AtomicBool::new(false),
            detector_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) -> SklResult<()> {
        self.is_running.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub async fn stop(&self) -> SklResult<()> {
        self.is_running.store(false, Ordering::SeqCst);
        Ok(())
    }

    pub fn detect_anomaly(&self, check_name: &str, value: f64) -> Option<Anomaly> {
        let mut baseline_data = self.baseline_data.write().unwrap();
        let baseline = baseline_data.entry(check_name.to_string()).or_insert_with(BaselineData::new);

        baseline.add_value(value);

        if baseline.has_sufficient_data() {
            let z_score = baseline.calculate_z_score(value);

            if z_score.abs() > 2.0 {  // 2 standard deviations
                let anomaly = Anomaly {
                    check_name: check_name.to_string(),
                    value,
                    expected_value: baseline.mean(),
                    deviation: z_score,
                    timestamp: Instant::now(),
                    severity: if z_score.abs() > 3.0 {
                        AnomalySeverity::High
                    } else {
                        AnomalySeverity::Medium
                    },
                };

                let mut anomalies = self.anomalies.write().unwrap();
                anomalies.push_back(anomaly.clone());

                if anomalies.len() > 1000 {
                    anomalies.pop_front();
                }

                return Some(anomaly);
            }
        }

        None
    }

    pub async fn get_recent_anomalies(&self) -> SklResult<Vec<Anomaly>> {
        let anomalies = self.anomalies.read().unwrap();
        Ok(anomalies.iter().cloned().collect())
    }
}

/// Baseline data for anomaly detection
#[derive(Debug)]
pub struct BaselineData {
    pub values: VecDeque<f64>,
    pub sum: f64,
    pub sum_squares: f64,
    pub count: usize,
    pub max_values: usize,
}

impl BaselineData {
    pub fn new() -> Self {
        Self {
            values: VecDeque::new(),
            sum: 0.0,
            sum_squares: 0.0,
            count: 0,
            max_values: 100,
        }
    }

    pub fn add_value(&mut self, value: f64) {
        if self.values.len() >= self.max_values {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
                self.sum_squares -= old_value * old_value;
                self.count -= 1;
            }
        }

        self.values.push_back(value);
        self.sum += value;
        self.sum_squares += value * value;
        self.count += 1;
    }

    pub fn has_sufficient_data(&self) -> bool {
        self.count >= 10
    }

    pub fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    pub fn standard_deviation(&self) -> f64 {
        if self.count > 1 {
            let mean = self.mean();
            let variance = (self.sum_squares - self.count as f64 * mean * mean) / (self.count - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        }
    }

    pub fn calculate_z_score(&self, value: f64) -> f64 {
        let mean = self.mean();
        let std_dev = self.standard_deviation();

        if std_dev > 0.0 {
            (value - mean) / std_dev
        } else {
            0.0
        }
    }
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub check_name: String,
    pub value: f64,
    pub expected_value: f64,
    pub deviation: f64,
    pub timestamp: Instant,
    pub severity: AnomalySeverity,
}

/// Anomaly severity
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Health alert manager
#[derive(Debug)]
pub struct HealthAlertManager {
    pub alert_rules: Arc<RwLock<Vec<HealthAlertRule>>>,
    pub active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,
    pub alert_history: Arc<RwLock<VecDeque<HealthAlert>>>,
    pub is_running: AtomicBool,
    pub manager_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl HealthAlertManager {
    pub fn new() -> Self {
        Self {
            alert_rules: Arc::new(RwLock::new(vec![])),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            is_running: AtomicBool::new(false),
            manager_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) -> SklResult<()> {
        self.is_running.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub async fn stop(&self) -> SklResult<()> {
        self.is_running.store(false, Ordering::SeqCst);
        Ok(())
    }

    pub fn add_alert_rule(&self, rule: HealthAlertRule) {
        let mut rules = self.alert_rules.write().unwrap();
        rules.push(rule);
    }

    pub async fn check_result(&self, result: &HealthCheckResult) -> SklResult<()> {
        let rules = self.alert_rules.read().unwrap();

        for rule in rules.iter() {
            if rule.matches_result(result) {
                self.trigger_alert(rule, result).await?;
            }
        }

        Ok(())
    }

    pub async fn get_active_alerts(&self) -> SklResult<Vec<ActiveAlert>> {
        let alerts = self.active_alerts.read().unwrap();
        Ok(alerts.values().cloned().collect())
    }

    async fn trigger_alert(&self, rule: &HealthAlertRule, result: &HealthCheckResult) -> SklResult<()> {
        let alert_id = format!("{}_{}", rule.name, result.timestamp.elapsed().as_millis());

        let alert = HealthAlert {
            alert_id: alert_id.clone(),
            check_name: result.check_name.clone(),
            severity: rule.severity.clone(),
            message: rule.generate_message(result),
            details: result.details.clone(),
            timestamp: result.timestamp,
            resolved: false,
        };

        // Add to active alerts
        let mut active_alerts = self.active_alerts.write().unwrap();
        active_alerts.insert(alert_id.clone(), ActiveAlert {
            alert: alert.clone(),
            rule_name: rule.name.clone(),
            acknowledgments: vec![],
        });

        // Add to history
        let mut history = self.alert_history.write().unwrap();
        history.push_back(alert);
        if history.len() > 1000 {
            history.pop_front();
        }

        Ok(())
    }
}

/// Health alert rule
#[derive(Debug, Clone)]
pub struct HealthAlertRule {
    pub name: String,
    pub check_pattern: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub cooldown: Duration,
    pub enabled: bool,
}

impl HealthAlertRule {
    pub fn matches_result(&self, result: &HealthCheckResult) -> bool {
        // Simple pattern matching - in a real implementation, this would be more sophisticated
        let matches_pattern = self.check_pattern == "*" || result.check_name.contains(&self.check_pattern);
        let matches_condition = self.condition.evaluate(result);

        matches_pattern && matches_condition && self.enabled
    }

    pub fn generate_message(&self, result: &HealthCheckResult) -> String {
        format!("Alert: {} - {}", result.check_name, result.message)
    }
}

/// Alert condition
#[derive(Debug, Clone)]
pub struct AlertCondition {
    pub field: String,
    pub operator: ComparisonOperator,
    pub value: f64,
}

impl AlertCondition {
    pub fn evaluate(&self, result: &HealthCheckResult) -> bool {
        let field_value = match self.field.as_str() {
            "healthy" => if result.healthy { 1.0 } else { 0.0 },
            "response_time_ms" => result.response_time.as_millis() as f64,
            _ => {
                // Check custom metrics
                result.metrics.get(&self.field).copied().unwrap_or(0.0)
            }
        };

        match self.operator {
            ComparisonOperator::Equals => field_value == self.value,
            ComparisonOperator::NotEquals => field_value != self.value,
            ComparisonOperator::GreaterThan => field_value > self.value,
            ComparisonOperator::GreaterThanOrEqual => field_value >= self.value,
            ComparisonOperator::LessThan => field_value < self.value,
            ComparisonOperator::LessThanOrEqual => field_value <= self.value,
        }
    }
}

/// Comparison operator
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    Equals,
    NotEquals,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
}

/// Active alert
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    pub alert: HealthAlert,
    pub rule_name: String,
    pub acknowledgments: Vec<AlertAcknowledgment>,
}

/// Alert acknowledgment
#[derive(Debug, Clone)]
pub struct AlertAcknowledgment {
    pub user: String,
    pub timestamp: Instant,
    pub message: String,
}

/// Auto recovery manager
#[derive(Debug)]
pub struct AutoRecoveryManager {
    pub recovery_strategies: Arc<RwLock<HashMap<String, RecoveryStrategy>>>,
    pub active_recoveries: Arc<RwLock<HashMap<String, ActiveRecovery>>>,
    pub is_running: AtomicBool,
    pub manager_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl AutoRecoveryManager {
    pub fn new() -> Self {
        Self {
            recovery_strategies: Arc::new(RwLock::new(HashMap::new())),
            active_recoveries: Arc::new(RwLock::new(HashMap::new())),
            is_running: AtomicBool::new(false),
            manager_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) -> SklResult<()> {
        self.is_running.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub async fn stop(&self) -> SklResult<()> {
        self.is_running.store(false, Ordering::SeqCst);
        Ok(())
    }

    pub fn add_recovery_strategy(&self, check_name: String, strategy: RecoveryStrategy) {
        let mut strategies = self.recovery_strategies.write().unwrap();
        strategies.insert(check_name, strategy);
    }

    pub async fn trigger_recovery(&self, check_name: &str) -> SklResult<()> {
        let strategies = self.recovery_strategies.read().unwrap();

        if let Some(strategy) = strategies.get(check_name) {
            let recovery_id = format!("{}_{}", check_name, Instant::now().elapsed().as_millis());

            let active_recovery = ActiveRecovery {
                recovery_id: recovery_id.clone(),
                check_name: check_name.to_string(),
                strategy: strategy.clone(),
                status: RecoveryStatus::InProgress,
                started_at: Instant::now(),
                completed_at: None,
                attempts: 0,
                max_attempts: strategy.max_attempts,
            };

            let mut active_recoveries = self.active_recoveries.write().unwrap();
            active_recoveries.insert(recovery_id, active_recovery);
        }

        Ok(())
    }
}

/// Recovery strategy
#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    pub name: String,
    pub recovery_type: RecoveryType,
    pub max_attempts: usize,
    pub retry_delay: Duration,
    pub timeout: Duration,
}

/// Recovery type
#[derive(Debug, Clone)]
pub enum RecoveryType {
    RestartService,
    ClearCache,
    ResetConnection,
    Custom(String),
}

/// Active recovery
#[derive(Debug, Clone)]
pub struct ActiveRecovery {
    pub recovery_id: String,
    pub check_name: String,
    pub strategy: RecoveryStrategy,
    pub status: RecoveryStatus,
    pub started_at: Instant,
    pub completed_at: Option<Instant>,
    pub attempts: usize,
    pub max_attempts: usize,
}

/// Recovery status
#[derive(Debug, Clone)]
pub enum RecoveryStatus {
    InProgress,
    Success,
    Failed,
    Timeout,
}

/// Health metrics collector
#[derive(Debug)]
pub struct HealthMetricsCollector {
    pub metrics: Arc<RwLock<HealthMetrics>>,
}

impl HealthMetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HealthMetrics::new())),
        }
    }

    pub fn record_check_execution(&self, duration: Duration, success: bool) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_checks += 1;
        if success {
            metrics.successful_checks += 1;
        } else {
            metrics.failed_checks += 1;
        }
        metrics.total_execution_time += duration;
    }

    pub fn get_metrics(&self) -> HealthMetrics {
        self.metrics.read().unwrap().clone()
    }
}

/// Health metrics
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    pub total_checks: usize,
    pub successful_checks: usize,
    pub failed_checks: usize,
    pub total_execution_time: Duration,
    pub alerts_triggered: usize,
    pub recoveries_attempted: usize,
    pub recoveries_successful: usize,
}

impl HealthMetrics {
    pub fn new() -> Self {
        Self {
            total_checks: 0,
            successful_checks: 0,
            failed_checks: 0,
            total_execution_time: Duration::from_millis(0),
            alerts_triggered: 0,
            recoveries_attempted: 0,
            recoveries_successful: 0,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_checks > 0 {
            self.successful_checks as f64 / self.total_checks as f64
        } else {
            0.0
        }
    }

    pub fn average_execution_time(&self) -> Duration {
        if self.total_checks > 0 {
            self.total_execution_time / self.total_checks as u32
        } else {
            Duration::from_millis(0)
        }
    }
}

/// Health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub summary: HealthSummary,
    pub trends: HashMap<String, TrendInfo>,
    pub anomalies: Vec<Anomaly>,
    pub alerts: Vec<ActiveAlert>,
    pub timestamp: Instant,
}