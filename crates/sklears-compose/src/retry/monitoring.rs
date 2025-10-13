//! Monitoring and Metrics for Retry Systems
//!
//! This module provides comprehensive monitoring, metrics collection, alerting,
//! and performance tracking capabilities for retry management systems with
//! real-time analytics and adaptive optimization features.

use super::core::*;
use super::simd_operations::*;
use sklears_core::error::Result as SklResult;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, RwLock},
    time::{Duration, SystemTime},
};

/// Retry metrics collector
#[derive(Debug)]
pub struct RetryMetricsCollector {
    /// Metrics storage
    metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    /// Metrics history
    history: Arc<Mutex<VecDeque<MetricSnapshot>>>,
    /// Metrics configuration
    config: MetricsConfig,
    /// Publishers
    publishers: Vec<Box<dyn MetricsPublisher + Send + Sync>>,
    /// Aggregators
    aggregators: HashMap<String, Box<dyn MetricsAggregator + Send + Sync>>,
}

impl RetryMetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        let mut collector = Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(Mutex::new(VecDeque::new())),
            config: MetricsConfig::default(),
            publishers: Vec::new(),
            aggregators: HashMap::new(),
        };

        // Register default aggregators
        collector.register_aggregator("success_rate", Box::new(SuccessRateAggregator::new()));
        collector.register_aggregator("duration", Box::new(DurationAggregator::new()));
        collector.register_aggregator("error_rate", Box::new(ErrorRateAggregator::new()));

        collector
    }

    /// Register metrics publisher
    pub fn register_publisher(&mut self, publisher: Box<dyn MetricsPublisher + Send + Sync>) {
        self.publishers.push(publisher);
    }

    /// Register metrics aggregator
    pub fn register_aggregator(&mut self, name: &str, aggregator: Box<dyn MetricsAggregator + Send + Sync>) {
        self.aggregators.insert(name.to_string(), aggregator);
    }

    /// Record retry metrics
    pub fn record_retry_metrics(&self, metrics: &RetryMetrics) -> SklResult<()> {
        let timestamp = SystemTime::now();

        // Update individual metrics
        {
            let mut metrics_map = self.metrics.write().unwrap();

            metrics_map.insert("total_attempts".to_string(), MetricValue::Counter(metrics.total_attempts));
            metrics_map.insert("successful_attempts".to_string(), MetricValue::Counter(metrics.successful_attempts));
            metrics_map.insert("failed_attempts".to_string(), MetricValue::Counter(metrics.failed_attempts));
            metrics_map.insert("success_rate".to_string(), MetricValue::Gauge(metrics.success_rate));
            metrics_map.insert("avg_duration_ms".to_string(), MetricValue::Gauge(metrics.avg_duration.as_millis() as f64));
            metrics_map.insert("total_duration_ms".to_string(), MetricValue::Counter(metrics.total_duration.as_millis()));
        }

        // Create snapshot
        let snapshot = MetricSnapshot {
            timestamp,
            metrics: metrics.clone(),
            aggregated_metrics: self.calculate_aggregated_metrics(),
        };

        // Add to history
        {
            let mut history = self.history.lock().unwrap();
            history.push_back(snapshot);

            // Limit history size
            if history.len() > 10000 {
                history.pop_front();
            }
        }

        // Publish metrics
        for publisher in &self.publishers {
            if let Err(e) = publisher.publish(metrics) {
                eprintln!("Failed to publish metrics: {:?}", e);
            }
        }

        Ok(())
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> HashMap<String, MetricValue> {
        self.metrics.read().unwrap().clone()
    }

    /// Get metrics history
    pub fn get_history(&self, duration: Duration) -> Vec<MetricSnapshot> {
        let cutoff_time = SystemTime::now() - duration;
        let history = self.history.lock().unwrap();

        history.iter()
            .filter(|snapshot| snapshot.timestamp >= cutoff_time)
            .cloned()
            .collect()
    }

    /// Calculate aggregated metrics
    fn calculate_aggregated_metrics(&self) -> HashMap<String, f64> {
        let mut aggregated = HashMap::new();

        let history = self.history.lock().unwrap();
        let recent_snapshots: Vec<_> = history.iter().rev().take(10).collect();

        for (name, aggregator) in &self.aggregators {
            let metrics: Vec<&RetryMetrics> = recent_snapshots.iter().map(|s| &s.metrics).collect();
            let aggregated_value = aggregator.aggregate(&metrics);
            aggregated.insert(name.clone(), aggregated_value);
        }

        aggregated
    }

    /// Get performance statistics
    pub fn get_performance_statistics(&self) -> PerformanceStatistics {
        let history = self.history.lock().unwrap();

        if history.is_empty() {
            return PerformanceStatistics::default();
        }

        let recent_snapshots: Vec<_> = history.iter().rev().take(100).collect();
        let total_attempts: u64 = recent_snapshots.iter().map(|s| s.metrics.total_attempts).sum();
        let successful_attempts: u64 = recent_snapshots.iter().map(|s| s.metrics.successful_attempts).sum();
        let total_duration: Duration = recent_snapshots.iter().map(|s| s.metrics.total_duration).sum();

        let overall_success_rate = if total_attempts > 0 {
            successful_attempts as f64 / total_attempts as f64
        } else {
            0.0
        };

        let avg_duration = if total_attempts > 0 {
            total_duration / total_attempts as u32
        } else {
            Duration::ZERO
        };

        PerformanceStatistics {
            total_attempts,
            successful_attempts,
            overall_success_rate,
            avg_duration,
            uptime: SystemTime::now().duration_since(
                history.front().map(|s| s.timestamp).unwrap_or_else(SystemTime::now)
            ).unwrap_or(Duration::ZERO),
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Collection interval
    pub collection_interval: Duration,
    /// Retention period
    pub retention_period: Duration,
    /// Metrics to collect
    pub metrics_filter: Vec<String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_secs(60),
            retention_period: Duration::from_secs(86400), // 24 hours
            metrics_filter: vec![
                "success_rate".to_string(),
                "avg_duration".to_string(),
                "total_attempts".to_string(),
                "error_rates".to_string(),
            ],
        }
    }
}

/// Retry metrics data structure
#[derive(Debug, Clone)]
pub struct RetryMetrics {
    /// Total retry attempts
    pub total_attempts: u64,
    /// Successful attempts
    pub successful_attempts: u64,
    /// Failed attempts
    pub failed_attempts: u64,
    /// Success rate
    pub success_rate: f64,
    /// Average duration
    pub avg_duration: Duration,
    /// Total duration
    pub total_duration: Duration,
    /// Error rates by type
    pub error_rates: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Metric value types
#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Timer(Duration),
}

/// Metric snapshot
#[derive(Debug, Clone)]
pub struct MetricSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Raw metrics
    pub metrics: RetryMetrics,
    /// Aggregated metrics
    pub aggregated_metrics: HashMap<String, f64>,
}

/// Performance statistics
#[derive(Debug, Clone, Default)]
pub struct PerformanceStatistics {
    /// Total attempts across all operations
    pub total_attempts: u64,
    /// Total successful attempts
    pub successful_attempts: u64,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Average duration per attempt
    pub avg_duration: Duration,
    /// System uptime
    pub uptime: Duration,
}

/// Metrics publisher trait
pub trait MetricsPublisher: Send + Sync {
    /// Publish metrics
    fn publish(&self, metrics: &RetryMetrics) -> SklResult<()>;

    /// Get publisher name
    fn name(&self) -> &str;
}

/// Console metrics publisher
#[derive(Debug)]
pub struct ConsolePublisher {
    /// Publication interval
    pub interval: Duration,
    /// Last publication time
    last_published: Arc<Mutex<Option<SystemTime>>>,
}

impl ConsolePublisher {
    /// Create new console publisher
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_published: Arc::new(Mutex::new(None)),
        }
    }
}

impl MetricsPublisher for ConsolePublisher {
    fn publish(&self, metrics: &RetryMetrics) -> SklResult<()> {
        let mut last_published = self.last_published.lock().unwrap();
        let now = SystemTime::now();

        // Check if enough time has passed
        if let Some(last_time) = *last_published {
            if now.duration_since(last_time).unwrap_or(Duration::ZERO) < self.interval {
                return Ok(());
            }
        }

        println!("=== Retry Metrics ===");
        println!("Total attempts: {}", metrics.total_attempts);
        println!("Success rate: {:.2}%", metrics.success_rate * 100.0);
        println!("Average duration: {:?}", metrics.avg_duration);
        println!("Error rates:");
        for (error_type, rate) in &metrics.error_rates {
            println!("  {}: {:.2}%", error_type, rate * 100.0);
        }
        println!("=====================");

        *last_published = Some(now);
        Ok(())
    }

    fn name(&self) -> &str {
        "console"
    }
}

/// JSON file publisher
#[derive(Debug)]
pub struct JsonFilePublisher {
    /// File path
    pub file_path: String,
    /// Append mode
    pub append: bool,
}

impl JsonFilePublisher {
    /// Create new JSON file publisher
    pub fn new(file_path: String, append: bool) -> Self {
        Self { file_path, append }
    }
}

impl MetricsPublisher for JsonFilePublisher {
    fn publish(&self, metrics: &RetryMetrics) -> SklResult<()> {
        // In a real implementation, this would serialize metrics to JSON and write to file
        // Simplified for this example
        println!("Publishing metrics to JSON file: {}", self.file_path);
        Ok(())
    }

    fn name(&self) -> &str {
        "json_file"
    }
}

/// Metrics aggregator trait
pub trait MetricsAggregator: Send + Sync {
    /// Aggregate metrics
    fn aggregate(&self, metrics: &[&RetryMetrics]) -> f64;

    /// Get aggregator name
    fn name(&self) -> &str;
}

/// Success rate aggregator
#[derive(Debug)]
pub struct SuccessRateAggregator {
    /// Aggregation method
    method: AggregationMethod,
}

impl SuccessRateAggregator {
    /// Create new success rate aggregator
    pub fn new() -> Self {
        Self {
            method: AggregationMethod::Average,
        }
    }

    /// Set aggregation method
    pub fn with_method(mut self, method: AggregationMethod) -> Self {
        self.method = method;
        self
    }
}

impl MetricsAggregator for SuccessRateAggregator {
    fn aggregate(&self, metrics: &[&RetryMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let success_rates: Vec<f64> = metrics.iter().map(|m| m.success_rate).collect();

        match self.method {
            AggregationMethod::Average => success_rates.iter().sum::<f64>() / success_rates.len() as f64,
            AggregationMethod::Median => {
                let mut sorted = success_rates.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            }
            AggregationMethod::Maximum => success_rates.iter().fold(0.0, |acc, &x| acc.max(x)),
            AggregationMethod::Minimum => success_rates.iter().fold(1.0, |acc, &x| acc.min(x)),
        }
    }

    fn name(&self) -> &str {
        "success_rate"
    }
}

/// Duration aggregator
#[derive(Debug)]
pub struct DurationAggregator {
    /// Aggregation method
    method: AggregationMethod,
}

impl DurationAggregator {
    /// Create new duration aggregator
    pub fn new() -> Self {
        Self {
            method: AggregationMethod::Average,
        }
    }
}

impl MetricsAggregator for DurationAggregator {
    fn aggregate(&self, metrics: &[&RetryMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let durations: Vec<f64> = metrics.iter().map(|m| m.avg_duration.as_millis() as f64).collect();

        match self.method {
            AggregationMethod::Average => durations.iter().sum::<f64>() / durations.len() as f64,
            AggregationMethod::Median => {
                let mut sorted = durations.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            }
            AggregationMethod::Maximum => durations.iter().fold(0.0, |acc, &x| acc.max(x)),
            AggregationMethod::Minimum => durations.iter().fold(f64::INFINITY, |acc, &x| acc.min(x)),
        }
    }

    fn name(&self) -> &str {
        "duration"
    }
}

/// Error rate aggregator
#[derive(Debug)]
pub struct ErrorRateAggregator;

impl ErrorRateAggregator {
    /// Create new error rate aggregator
    pub fn new() -> Self {
        Self
    }
}

impl MetricsAggregator for ErrorRateAggregator {
    fn aggregate(&self, metrics: &[&RetryMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let total_attempts: u64 = metrics.iter().map(|m| m.total_attempts).sum();
        let failed_attempts: u64 = metrics.iter().map(|m| m.failed_attempts).sum();

        if total_attempts > 0 {
            failed_attempts as f64 / total_attempts as f64
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "error_rate"
    }
}

/// Aggregation method enumeration
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    Average,
    Median,
    Maximum,
    Minimum,
}

/// Alerting system
#[derive(Debug)]
pub struct AlertingSystem {
    /// Alert rules
    rules: Vec<AlertRule>,
    /// Alert handlers
    handlers: Vec<Box<dyn AlertHandler + Send + Sync>>,
    /// Alert history
    history: Arc<Mutex<VecDeque<Alert>>>,
    /// Alerting configuration
    config: AlertingConfig,
}

impl AlertingSystem {
    /// Create new alerting system
    pub fn new() -> Self {
        let mut system = Self {
            rules: Vec::new(),
            handlers: Vec::new(),
            history: Arc::new(Mutex::new(VecDeque::new())),
            config: AlertingConfig::default(),
        };

        // Add default alert rules
        system.add_rule(AlertRule {
            name: "low_success_rate".to_string(),
            threshold: 0.5,
            operator: ComparisonOperator::LessThan,
            severity: AlertSeverity::Warning,
            metric: "success_rate".to_string(),
            enabled: true,
        });

        system.add_rule(AlertRule {
            name: "high_error_rate".to_string(),
            threshold: 0.3,
            operator: ComparisonOperator::GreaterThan,
            severity: AlertSeverity::Error,
            metric: "error_rate".to_string(),
            enabled: true,
        });

        system
    }

    /// Add alert rule
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Register alert handler
    pub fn register_handler(&mut self, handler: Box<dyn AlertHandler + Send + Sync>) {
        self.handlers.push(handler);
    }

    /// Check metrics against alert rules
    pub fn check_alerts(&self, metrics: &RetryMetrics) -> SklResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let metric_values = self.extract_metric_values(metrics);

        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }

            if let Some(&metric_value) = metric_values.get(&rule.metric) {
                let should_alert = match rule.operator {
                    ComparisonOperator::LessThan => metric_value < rule.threshold,
                    ComparisonOperator::LessThanOrEqual => metric_value <= rule.threshold,
                    ComparisonOperator::GreaterThan => metric_value > rule.threshold,
                    ComparisonOperator::GreaterThanOrEqual => metric_value >= rule.threshold,
                    ComparisonOperator::Equal => (metric_value - rule.threshold).abs() < 0.001,
                    ComparisonOperator::NotEqual => (metric_value - rule.threshold).abs() >= 0.001,
                };

                if should_alert {
                    let alert = Alert {
                        id: format!("{}_{}", rule.name, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis()),
                        timestamp: SystemTime::now(),
                        severity: rule.severity.clone(),
                        message: format!("{} alert: {} {} {}", rule.severity.to_string(), rule.metric, rule.operator.to_string(), rule.threshold),
                        metric_name: rule.metric.clone(),
                        metric_value,
                        rule_name: rule.name.clone(),
                        metadata: HashMap::new(),
                    };

                    self.trigger_alert(alert)?;
                }
            }
        }

        Ok(())
    }

    /// Trigger alert
    fn trigger_alert(&self, alert: Alert) -> SklResult<()> {
        // Add to history
        {
            let mut history = self.history.lock().unwrap();
            history.push_back(alert.clone());

            // Limit history size
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        // Send to handlers
        for handler in &self.handlers {
            if let Err(e) = handler.handle_alert(&alert) {
                eprintln!("Alert handler {} failed: {:?}", handler.name(), e);
            }
        }

        Ok(())
    }

    /// Extract metric values for alerting
    fn extract_metric_values(&self, metrics: &RetryMetrics) -> HashMap<String, f64> {
        let mut values = HashMap::new();
        values.insert("success_rate".to_string(), metrics.success_rate);
        values.insert("error_rate".to_string(), 1.0 - metrics.success_rate);
        values.insert("avg_duration_ms".to_string(), metrics.avg_duration.as_millis() as f64);
        values.insert("total_attempts".to_string(), metrics.total_attempts as f64);
        values
    }

    /// Get alert history
    pub fn get_alert_history(&self, duration: Duration) -> Vec<Alert> {
        let cutoff_time = SystemTime::now() - duration;
        let history = self.history.lock().unwrap();

        history.iter()
            .filter(|alert| alert.timestamp >= cutoff_time)
            .cloned()
            .collect()
    }
}

/// Alert rule
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Metric name to monitor
    pub metric: String,
    /// Alert threshold
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Rule enabled
    pub enabled: bool,
}

/// Comparison operator enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

impl std::fmt::Display for ComparisonOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonOperator::LessThan => write!(f, "<"),
            ComparisonOperator::LessThanOrEqual => write!(f, "<="),
            ComparisonOperator::GreaterThan => write!(f, ">"),
            ComparisonOperator::GreaterThanOrEqual => write!(f, ">="),
            ComparisonOperator::Equal => write!(f, "=="),
            ComparisonOperator::NotEqual => write!(f, "!="),
        }
    }
}

/// Alert severity enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Error => write!(f, "ERROR"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Alert data structure
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert identifier
    pub id: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Metric name that triggered alert
    pub metric_name: String,
    /// Metric value that triggered alert
    pub metric_value: f64,
    /// Rule name that triggered alert
    pub rule_name: String,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

/// Alert handler trait
pub trait AlertHandler: Send + Sync {
    /// Handle alert
    fn handle_alert(&self, alert: &Alert) -> SklResult<()>;

    /// Get handler name
    fn name(&self) -> &str;
}

/// Console alert handler
#[derive(Debug)]
pub struct ConsoleAlertHandler;

impl AlertHandler for ConsoleAlertHandler {
    fn handle_alert(&self, alert: &Alert) -> SklResult<()> {
        println!("🚨 ALERT [{}] {}: {}", alert.severity, alert.rule_name, alert.message);
        Ok(())
    }

    fn name(&self) -> &str {
        "console"
    }
}

/// Email alert handler (stub implementation)
#[derive(Debug)]
pub struct EmailAlertHandler {
    /// Email recipients
    pub recipients: Vec<String>,
}

impl EmailAlertHandler {
    /// Create new email alert handler
    pub fn new(recipients: Vec<String>) -> Self {
        Self { recipients }
    }
}

impl AlertHandler for EmailAlertHandler {
    fn handle_alert(&self, alert: &Alert) -> SklResult<()> {
        // In a real implementation, this would send emails
        println!("Sending email alert to {:?}: {}", self.recipients, alert.message);
        Ok(())
    }

    fn name(&self) -> &str {
        "email"
    }
}

/// Alerting configuration
#[derive(Debug, Clone)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert check interval
    pub check_interval: Duration,
    /// Alert cooldown period
    pub cooldown_period: Duration,
    /// Maximum alerts per period
    pub max_alerts_per_period: usize,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval: Duration::from_secs(60),
            cooldown_period: Duration::from_secs(300), // 5 minutes
            max_alerts_per_period: 10,
        }
    }
}

/// Monitoring factory
pub struct MonitoringFactory;

impl MonitoringFactory {
    /// Create complete monitoring system
    pub fn create_monitoring_system(config: &HashMap<String, String>) -> (RetryMetricsCollector, AlertingSystem) {
        let mut metrics_collector = RetryMetricsCollector::new();
        let mut alerting_system = AlertingSystem::new();

        // Configure publishers
        if config.get("console_publisher").map(|s| s == "true").unwrap_or(false) {
            let interval = config.get("console_interval")
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(Duration::from_secs(30));
            metrics_collector.register_publisher(Box::new(ConsolePublisher::new(interval)));
        }

        if let Some(json_path) = config.get("json_file") {
            let append = config.get("json_append").map(|s| s == "true").unwrap_or(true);
            metrics_collector.register_publisher(Box::new(JsonFilePublisher::new(json_path.clone(), append)));
        }

        // Configure alert handlers
        if config.get("console_alerts").map(|s| s == "true").unwrap_or(true) {
            alerting_system.register_handler(Box::new(ConsoleAlertHandler));
        }

        if let Some(email_list) = config.get("email_alerts") {
            let recipients: Vec<String> = email_list.split(',').map(|s| s.trim().to_string()).collect();
            alerting_system.register_handler(Box::new(EmailAlertHandler::new(recipients)));
        }

        (metrics_collector, alerting_system)
    }

    /// Create default monitoring system
    pub fn create_default() -> (RetryMetricsCollector, AlertingSystem) {
        let mut metrics_collector = RetryMetricsCollector::new();
        let mut alerting_system = AlertingSystem::new();

        // Default console publisher
        metrics_collector.register_publisher(Box::new(ConsolePublisher::new(Duration::from_secs(60))));

        // Default console alert handler
        alerting_system.register_handler(Box::new(ConsoleAlertHandler));

        (metrics_collector, alerting_system)
    }
}