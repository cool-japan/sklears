//! Automated Alerting System for Pipeline Monitoring
//!
//! This module provides comprehensive alerting capabilities for pipeline monitoring,
//! including real-time anomaly detection, threshold-based alerts, and integration
//! with external notification systems.

use crate::monitoring::{Anomaly, AnomalySeverity, Metric};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Automated alerting system for pipeline monitoring
pub struct AutomatedAlerter {
    /// Alert configuration
    config: AlertConfig,
    /// Alert rules
    rules: RwLock<HashMap<String, AlertRule>>,
    /// Alert channels
    channels: RwLock<HashMap<String, Box<dyn AlertChannel>>>,
    /// Alert history
    alert_history: Arc<Mutex<VecDeque<AlertEvent>>>,
    /// Silenced alerts
    silenced_alerts: RwLock<HashMap<String, SilencePeriod>>,
    /// Active alerts
    active_alerts: RwLock<HashMap<String, ActiveAlert>>,
    /// Statistics
    stats: RwLock<AlertStats>,
}

/// Alert system configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Maximum alert history to retain
    pub max_history: usize,
    /// Default alert cooldown period
    pub default_cooldown: Duration,
    /// Enable alert grouping
    pub enable_grouping: bool,
    /// Alert grouping window
    pub grouping_window: Duration,
    /// Maximum alerts per grouping window
    pub max_alerts_per_group: usize,
    /// Enable alert escalation
    pub enable_escalation: bool,
    /// Escalation levels
    pub escalation_levels: Vec<EscalationLevel>,
}

/// Alert escalation level
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Escalation trigger time
    pub trigger_after: Duration,
    /// Additional channels to notify
    pub channels: Vec<String>,
    /// Escalation severity threshold
    pub severity_threshold: AlertSeverity,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Target channels
    pub channels: Vec<String>,
    /// Cooldown period between alerts
    pub cooldown: Duration,
    /// Rule enabled status
    pub enabled: bool,
    /// Alert labels
    pub labels: HashMap<String, String>,
    /// Rule priority
    pub priority: u32,
}

/// Alert condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AlertCondition {
    /// Threshold-based condition
    Threshold {
        /// Metric name to monitor
        metric: String,
        /// Threshold operator
        operator: ThresholdOperator,
        /// Threshold value
        value: f64,
        /// Time window for evaluation
        window: Duration,
        /// Minimum data points required
        min_points: usize,
    },
    /// Rate-based condition
    Rate {
        /// Metric name to monitor
        metric: String,
        /// Rate threshold (per second)
        rate_threshold: f64,
        /// Time window for rate calculation
        window: Duration,
    },
    /// Anomaly detection condition
    Anomaly {
        /// Metric name to monitor
        metric: String,
        /// Anomaly detection sensitivity
        sensitivity: f64,
        /// Training window size
        training_window: Duration,
    },
    /// Composite condition
    Composite {
        /// Logical operator
        operator: LogicalOperator,
        /// Sub-conditions
        conditions: Vec<AlertCondition>,
    },
    /// Pattern-based condition
    Pattern {
        /// Pattern to match
        pattern: String,
        /// Field to search in
        field: PatternField,
        /// Case sensitive matching
        case_sensitive: bool,
    },
}

/// Threshold operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdOperator {
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
}

/// Logical operators for composite conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    /// All conditions must be true
    And,
    /// Any condition must be true
    Or,
    /// Condition must not be true
    Not,
}

/// Pattern field types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternField {
    /// Pipeline name
    PipelineName,
    /// Stage name
    StageName,
    /// Error message
    ErrorMessage,
    /// Log message
    LogMessage,
    /// Custom field
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
    /// Emergency alert
    Emergency,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
            AlertSeverity::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    /// Alert ID
    pub id: String,
    /// Rule ID that triggered this alert
    pub rule_id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,
    /// Associated pipeline
    pub pipeline: Option<String>,
    /// Associated stage
    pub stage: Option<String>,
    /// Alert labels
    pub labels: HashMap<String, String>,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
    /// Metric value that triggered the alert
    pub trigger_value: Option<f64>,
    /// Alert status
    pub status: AlertStatus,
}

/// Alert status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Alert is firing
    Firing,
    /// Alert has been acknowledged
    Acknowledged,
    /// Alert has been resolved
    Resolved,
    /// Alert has been silenced
    Silenced,
}

/// Active alert tracking
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// Original alert event
    pub event: AlertEvent,
    /// First firing time
    pub first_fired: DateTime<Utc>,
    /// Last update time
    pub last_updated: DateTime<Utc>,
    /// Fire count
    pub fire_count: u32,
    /// Escalation level
    pub escalation_level: usize,
    /// Channels notified
    pub channels_notified: Vec<String>,
}

/// Silence period for alerts
#[derive(Debug, Clone)]
pub struct SilencePeriod {
    /// Silence start time
    pub start: DateTime<Utc>,
    /// Silence end time
    pub end: DateTime<Utc>,
    /// Silence reason
    pub reason: String,
    /// User who created the silence
    pub created_by: String,
    /// Silenced rule patterns
    pub rule_patterns: Vec<String>,
}

/// Alert statistics
#[derive(Debug, Clone, Default)]
pub struct AlertStats {
    /// Total alerts fired
    pub total_alerts: u64,
    /// Alerts by severity
    pub alerts_by_severity: HashMap<AlertSeverity, u64>,
    /// Alerts by rule
    pub alerts_by_rule: HashMap<String, u64>,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Average resolution time
    pub avg_resolution_time: Duration,
    /// Current active alerts
    pub active_alert_count: usize,
}

/// Alert channel trait for different notification methods
pub trait AlertChannel: Send + Sync + std::fmt::Debug {
    /// Send an alert
    fn send_alert(&self, alert: &AlertEvent) -> SklResult<()>;

    /// Get channel name
    fn name(&self) -> &str;

    /// Check if channel is healthy
    fn health_check(&self) -> SklResult<()>;

    /// Get channel configuration
    fn config(&self) -> HashMap<String, String>;
}

/// Console alert channel for debugging
#[derive(Debug)]
pub struct ConsoleAlertChannel {
    /// Channel name
    name: String,
    /// Configuration
    config: HashMap<String, String>,
}

/// Email alert channel
#[derive(Debug)]
pub struct EmailAlertChannel {
    /// Channel name
    name: String,
    /// SMTP server configuration
    smtp_server: String,
    /// SMTP port
    smtp_port: u16,
    /// Sender email
    from_email: String,
    /// Recipient emails
    to_emails: Vec<String>,
    /// Configuration
    config: HashMap<String, String>,
}

/// Webhook alert channel
#[derive(Debug)]
pub struct WebhookAlertChannel {
    /// Channel name
    name: String,
    /// Webhook URL
    url: String,
    /// HTTP headers
    headers: HashMap<String, String>,
    /// Request timeout
    timeout: Duration,
    /// Configuration
    config: HashMap<String, String>,
}

/// Slack alert channel
#[derive(Debug)]
pub struct SlackAlertChannel {
    /// Channel name
    name: String,
    /// Slack webhook URL
    webhook_url: String,
    /// Slack channel
    channel: String,
    /// Bot username
    username: String,
    /// Configuration
    config: HashMap<String, String>,
}

impl AutomatedAlerter {
    /// Create a new automated alerter
    #[must_use]
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            rules: RwLock::new(HashMap::new()),
            channels: RwLock::new(HashMap::new()),
            alert_history: Arc::new(Mutex::new(VecDeque::new())),
            silenced_alerts: RwLock::new(HashMap::new()),
            active_alerts: RwLock::new(HashMap::new()),
            stats: RwLock::new(AlertStats::default()),
        }
    }

    /// Add an alert rule
    pub fn add_rule(&self, rule: AlertRule) -> SklResult<()> {
        let mut rules = self.rules.write().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire write lock for rules".to_string())
        })?;

        rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Remove an alert rule
    pub fn remove_rule(&self, rule_id: &str) -> SklResult<()> {
        let mut rules = self.rules.write().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire write lock for rules".to_string())
        })?;

        rules.remove(rule_id);
        Ok(())
    }

    /// Add an alert channel
    pub fn add_channel(&self, name: &str, channel: Box<dyn AlertChannel>) -> SklResult<()> {
        let mut channels = self.channels.write().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire write lock for channels".to_string())
        })?;

        channels.insert(name.to_string(), channel);
        Ok(())
    }

    /// Remove an alert channel
    pub fn remove_channel(&self, name: &str) -> SklResult<()> {
        let mut channels = self.channels.write().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire write lock for channels".to_string())
        })?;

        channels.remove(name);
        Ok(())
    }

    /// Process metrics and check for alerts
    pub fn process_metrics(&self, metrics: &[Metric]) -> SklResult<Vec<AlertEvent>> {
        let mut triggered_alerts = Vec::new();

        let rules = self.rules.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire read lock for rules".to_string())
        })?;

        for (rule_id, rule) in rules.iter() {
            if !rule.enabled {
                continue;
            }

            // Check if rule is silenced
            if self.is_silenced(rule_id)? {
                continue;
            }

            // Check if rule is in cooldown
            if self.is_in_cooldown(rule_id)? {
                continue;
            }

            // Evaluate rule condition
            if self.evaluate_condition(&rule.condition, metrics)? {
                let alert = self.create_alert_event(rule, metrics)?;
                triggered_alerts.push(alert);
            }
        }

        // Process triggered alerts
        for alert in &triggered_alerts {
            self.handle_alert(alert.clone())?;
        }

        Ok(triggered_alerts)
    }

    /// Process anomalies from monitoring system
    pub fn process_anomalies(&self, anomalies: &[Anomaly]) -> SklResult<Vec<AlertEvent>> {
        let mut triggered_alerts = Vec::new();

        for anomaly in anomalies {
            // Create alert based on anomaly severity
            let alert_severity = match anomaly.severity {
                AnomalySeverity::Low => AlertSeverity::Info,
                AnomalySeverity::Medium => AlertSeverity::Warning,
                AnomalySeverity::High => AlertSeverity::Critical,
                AnomalySeverity::Critical => AlertSeverity::Emergency,
            };

            let alert_event = AlertEvent {
                id: uuid::Uuid::new_v4().to_string(),
                rule_id: "anomaly_detection".to_string(),
                severity: alert_severity,
                message: format!("Anomaly detected: {}", anomaly.description),
                timestamp: Utc::now(),
                pipeline: Some(anomaly.pipeline_name.clone()),
                stage: None,
                labels: {
                    let mut labels = HashMap::new();
                    labels.insert("type".to_string(), "anomaly".to_string());
                    labels.insert("metric".to_string(), anomaly.metric_name.clone());
                    labels
                },
                metadata: HashMap::new(),
                trigger_value: None,
                status: AlertStatus::Firing,
            };

            triggered_alerts.push(alert_event);
        }

        // Process triggered alerts
        for alert in &triggered_alerts {
            self.handle_alert(alert.clone())?;
        }

        Ok(triggered_alerts)
    }

    /// Create silence period
    pub fn create_silence(
        &self,
        rule_patterns: Vec<String>,
        duration: Duration,
        reason: String,
        created_by: String,
    ) -> SklResult<String> {
        let silence_id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();

        let silence = SilencePeriod {
            start: now,
            end: now
                + chrono::Duration::from_std(duration).map_err(|_| {
                    SklearsError::InvalidInput("Invalid duration for silence period".to_string())
                })?,
            reason,
            created_by,
            rule_patterns,
        };

        let mut silenced = self.silenced_alerts.write().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire write lock for silenced alerts".to_string(),
            )
        })?;

        silenced.insert(silence_id.clone(), silence);
        Ok(silence_id)
    }

    /// Remove silence period
    pub fn remove_silence(&self, silence_id: &str) -> SklResult<()> {
        let mut silenced = self.silenced_alerts.write().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire write lock for silenced alerts".to_string(),
            )
        })?;

        silenced.remove(silence_id);
        Ok(())
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&self, alert_id: &str, acknowledged_by: &str) -> SklResult<()> {
        let mut active_alerts = self.active_alerts.write().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire write lock for active alerts".to_string(),
            )
        })?;

        if let Some(active_alert) = active_alerts.get_mut(alert_id) {
            active_alert.event.status = AlertStatus::Acknowledged;
            active_alert
                .event
                .metadata
                .insert("acknowledged_by".to_string(), acknowledged_by.to_string());
            active_alert
                .event
                .metadata
                .insert("acknowledged_at".to_string(), Utc::now().to_rfc3339());
            active_alert.last_updated = Utc::now();
        }

        Ok(())
    }

    /// Resolve an alert
    pub fn resolve_alert(&self, alert_id: &str, resolved_by: &str) -> SklResult<()> {
        let mut active_alerts = self.active_alerts.write().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire write lock for active alerts".to_string(),
            )
        })?;

        if let Some(mut active_alert) = active_alerts.remove(alert_id) {
            active_alert.event.status = AlertStatus::Resolved;
            active_alert
                .event
                .metadata
                .insert("resolved_by".to_string(), resolved_by.to_string());
            active_alert
                .event
                .metadata
                .insert("resolved_at".to_string(), Utc::now().to_rfc3339());

            // Add to history (clone the event to avoid borrow issues)
            self.add_to_history(active_alert.event.clone())?;

            // Update statistics
            self.update_resolution_stats(&active_alert)?;
        }

        Ok(())
    }

    /// Get alert statistics
    pub fn get_stats(&self) -> SklResult<AlertStats> {
        let stats = self.stats.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire read lock for stats".to_string())
        })?;

        Ok(stats.clone())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> SklResult<Vec<ActiveAlert>> {
        let active_alerts = self.active_alerts.read().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire read lock for active alerts".to_string(),
            )
        })?;

        Ok(active_alerts.values().cloned().collect())
    }

    /// Get alert history
    pub fn get_alert_history(&self, limit: Option<usize>) -> SklResult<Vec<AlertEvent>> {
        let history = self.alert_history.lock().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire lock for alert history".to_string())
        })?;

        let alerts: Vec<AlertEvent> = history.iter().cloned().collect();

        if let Some(limit) = limit {
            Ok(alerts.into_iter().take(limit).collect())
        } else {
            Ok(alerts)
        }
    }

    /// Evaluate alert condition
    fn evaluate_condition(
        &self,
        condition: &AlertCondition,
        metrics: &[Metric],
    ) -> SklResult<bool> {
        match condition {
            AlertCondition::Threshold {
                metric,
                operator,
                value,
                window,
                min_points,
            } => self.evaluate_threshold_condition(
                metric,
                operator,
                *value,
                *window,
                *min_points,
                metrics,
            ),
            AlertCondition::Rate {
                metric,
                rate_threshold,
                window,
            } => self.evaluate_rate_condition(metric, *rate_threshold, *window, metrics),
            AlertCondition::Anomaly {
                metric,
                sensitivity,
                training_window,
            } => self.evaluate_anomaly_condition(metric, *sensitivity, *training_window, metrics),
            AlertCondition::Composite {
                operator,
                conditions,
            } => self.evaluate_composite_condition(operator, conditions, metrics),
            AlertCondition::Pattern {
                pattern,
                field,
                case_sensitive,
            } => self.evaluate_pattern_condition(pattern, field, *case_sensitive, metrics),
        }
    }

    /// Evaluate threshold condition
    fn evaluate_threshold_condition(
        &self,
        metric_name: &str,
        operator: &ThresholdOperator,
        threshold: f64,
        window: Duration,
        min_points: usize,
        metrics: &[Metric],
    ) -> SklResult<bool> {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - window.as_secs();

        let relevant_metrics: Vec<&Metric> = metrics
            .iter()
            .filter(|m| m.name == metric_name && m.timestamp >= cutoff_time)
            .collect();

        if relevant_metrics.len() < min_points {
            return Ok(false);
        }

        // Use the latest value for threshold comparison
        if let Some(latest_metric) = relevant_metrics.last() {
            Ok(self.compare_value(latest_metric.value, operator, threshold))
        } else {
            Ok(false)
        }
    }

    /// Evaluate rate condition
    fn evaluate_rate_condition(
        &self,
        metric_name: &str,
        rate_threshold: f64,
        window: Duration,
        metrics: &[Metric],
    ) -> SklResult<bool> {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - window.as_secs();

        let relevant_metrics: Vec<&Metric> = metrics
            .iter()
            .filter(|m| m.name == metric_name && m.timestamp >= cutoff_time)
            .collect();

        if relevant_metrics.len() < 2 {
            return Ok(false);
        }

        // Calculate rate of change
        let first = relevant_metrics.first().unwrap();
        let last = relevant_metrics.last().unwrap();

        let time_diff = last.timestamp - first.timestamp;
        if time_diff == 0 {
            return Ok(false);
        }

        let rate = (last.value - first.value) / time_diff as f64;
        Ok(rate > rate_threshold)
    }

    /// Evaluate anomaly condition
    fn evaluate_anomaly_condition(
        &self,
        metric_name: &str,
        sensitivity: f64,
        training_window: Duration,
        metrics: &[Metric],
    ) -> SklResult<bool> {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - training_window.as_secs();

        let relevant_metrics: Vec<&Metric> = metrics
            .iter()
            .filter(|m| m.name == metric_name && m.timestamp >= cutoff_time)
            .collect();

        if relevant_metrics.len() < 10 {
            return Ok(false);
        }

        // Simple anomaly detection using standard deviation
        let values: Vec<f64> = relevant_metrics.iter().map(|m| m.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if let Some(latest_metric) = relevant_metrics.last() {
            let z_score = (latest_metric.value - mean) / std_dev;
            Ok(z_score.abs() > sensitivity)
        } else {
            Ok(false)
        }
    }

    /// Evaluate composite condition
    fn evaluate_composite_condition(
        &self,
        operator: &LogicalOperator,
        conditions: &[AlertCondition],
        metrics: &[Metric],
    ) -> SklResult<bool> {
        match operator {
            LogicalOperator::And => {
                for condition in conditions {
                    if !self.evaluate_condition(condition, metrics)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            LogicalOperator::Or => {
                for condition in conditions {
                    if self.evaluate_condition(condition, metrics)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            LogicalOperator::Not => {
                if conditions.len() != 1 {
                    return Err(SklearsError::InvalidInput(
                        "NOT operator requires exactly one condition".to_string(),
                    ));
                }
                Ok(!self.evaluate_condition(&conditions[0], metrics)?)
            }
        }
    }

    /// Evaluate pattern condition
    fn evaluate_pattern_condition(
        &self,
        pattern: &str,
        field: &PatternField,
        case_sensitive: bool,
        metrics: &[Metric],
    ) -> SklResult<bool> {
        for metric in metrics {
            let text_to_search = match field {
                PatternField::PipelineName => &metric.pipeline_name,
                PatternField::StageName => {
                    if let Some(ref stage_name) = metric.stage_name {
                        stage_name
                    } else {
                        continue;
                    }
                }
                PatternField::ErrorMessage => {
                    // Would need error message field in Metric
                    continue;
                }
                PatternField::LogMessage => {
                    // Would need log message field in Metric
                    continue;
                }
                PatternField::Custom(field_name) => {
                    if let Some(value) = metric.metadata.get(field_name) {
                        value
                    } else {
                        continue;
                    }
                }
            };

            let matches = if case_sensitive {
                text_to_search.contains(pattern)
            } else {
                text_to_search
                    .to_lowercase()
                    .contains(&pattern.to_lowercase())
            };

            if matches {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Compare value using threshold operator
    fn compare_value(&self, value: f64, operator: &ThresholdOperator, threshold: f64) -> bool {
        match operator {
            ThresholdOperator::GreaterThan => value > threshold,
            ThresholdOperator::GreaterThanOrEqual => value >= threshold,
            ThresholdOperator::LessThan => value < threshold,
            ThresholdOperator::LessThanOrEqual => value <= threshold,
            ThresholdOperator::Equal => (value - threshold).abs() < f64::EPSILON,
            ThresholdOperator::NotEqual => (value - threshold).abs() >= f64::EPSILON,
        }
    }

    /// Create alert event from rule and metrics
    fn create_alert_event(&self, rule: &AlertRule, metrics: &[Metric]) -> SklResult<AlertEvent> {
        let trigger_value = metrics
            .iter()
            .find(|m| self.metric_matches_rule(m, rule))
            .map(|m| m.value);

        let pipeline = metrics
            .iter()
            .find(|m| self.metric_matches_rule(m, rule))
            .map(|m| m.pipeline_name.clone());

        let stage = metrics
            .iter()
            .find(|m| self.metric_matches_rule(m, rule))
            .and_then(|m| m.stage_name.clone());

        Ok(AlertEvent {
            id: uuid::Uuid::new_v4().to_string(),
            rule_id: rule.id.clone(),
            severity: rule.severity.clone(),
            message: format!("Alert triggered: {}", rule.description),
            timestamp: Utc::now(),
            pipeline,
            stage,
            labels: rule.labels.clone(),
            metadata: HashMap::new(),
            trigger_value,
            status: AlertStatus::Firing,
        })
    }

    /// Check if metric matches rule conditions
    fn metric_matches_rule(&self, _metric: &Metric, _rule: &AlertRule) -> bool {
        // Simplified implementation - would need more sophisticated matching
        true
    }

    /// Handle triggered alert
    fn handle_alert(&self, alert: AlertEvent) -> SklResult<()> {
        // Check if alert should be grouped
        if self.config.enable_grouping {
            if let Some(existing_alert_id) = self.find_groupable_alert(&alert)? {
                self.update_grouped_alert(existing_alert_id, &alert)?;
                return Ok(());
            }
        }

        // Add to active alerts
        let active_alert = ActiveAlert {
            event: alert.clone(),
            first_fired: alert.timestamp,
            last_updated: alert.timestamp,
            fire_count: 1,
            escalation_level: 0,
            channels_notified: Vec::new(),
        };

        {
            let mut active_alerts = self.active_alerts.write().map_err(|_| {
                SklearsError::InvalidOperation(
                    "Failed to acquire write lock for active alerts".to_string(),
                )
            })?;
            active_alerts.insert(alert.id.clone(), active_alert);
        }

        // Send notifications
        self.send_alert_notifications(&alert)?;

        // Update statistics
        self.update_stats(&alert)?;

        Ok(())
    }

    /// Send alert notifications to configured channels
    fn send_alert_notifications(&self, alert: &AlertEvent) -> SklResult<()> {
        let rules = self.rules.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire read lock for rules".to_string())
        })?;

        let channels = self.channels.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire read lock for channels".to_string())
        })?;

        if let Some(rule) = rules.get(&alert.rule_id) {
            for channel_name in &rule.channels {
                if let Some(channel) = channels.get(channel_name) {
                    if let Err(e) = channel.send_alert(alert) {
                        eprintln!("Failed to send alert to channel {channel_name}: {e:?}");
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if rule is currently silenced
    fn is_silenced(&self, rule_id: &str) -> SklResult<bool> {
        let silenced = self.silenced_alerts.read().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire read lock for silenced alerts".to_string(),
            )
        })?;

        let now = Utc::now();

        for silence in silenced.values() {
            if now >= silence.start && now <= silence.end {
                for pattern in &silence.rule_patterns {
                    if rule_id.contains(pattern) {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Check if rule is in cooldown period
    fn is_in_cooldown(&self, rule_id: &str) -> SklResult<bool> {
        let active_alerts = self.active_alerts.read().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire read lock for active alerts".to_string(),
            )
        })?;

        let rules = self.rules.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire read lock for rules".to_string())
        })?;

        if let Some(rule) = rules.get(rule_id) {
            let now = Utc::now();

            for active_alert in active_alerts.values() {
                if active_alert.event.rule_id == rule_id {
                    let cooldown_end = active_alert.last_updated
                        + chrono::Duration::from_std(rule.cooldown).unwrap_or_default();

                    if now < cooldown_end {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Find groupable alert
    fn find_groupable_alert(&self, alert: &AlertEvent) -> SklResult<Option<String>> {
        let active_alerts = self.active_alerts.read().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire read lock for active alerts".to_string(),
            )
        })?;

        let now = alert.timestamp;

        for (alert_id, active_alert) in active_alerts.iter() {
            let time_diff = now - active_alert.first_fired;
            let window_duration =
                chrono::Duration::from_std(self.config.grouping_window).unwrap_or_default();

            if time_diff <= window_duration
                && active_alert.event.rule_id == alert.rule_id
                && active_alert.event.pipeline == alert.pipeline
            {
                return Ok(Some(alert_id.clone()));
            }
        }

        Ok(None)
    }

    /// Update grouped alert
    fn update_grouped_alert(&self, alert_id: String, new_alert: &AlertEvent) -> SklResult<()> {
        let mut active_alerts = self.active_alerts.write().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire write lock for active alerts".to_string(),
            )
        })?;

        if let Some(active_alert) = active_alerts.get_mut(&alert_id) {
            active_alert.fire_count += 1;
            active_alert.last_updated = new_alert.timestamp;

            // Update severity if higher
            if new_alert.severity > active_alert.event.severity {
                active_alert.event.severity = new_alert.severity.clone();
            }
        }

        Ok(())
    }

    /// Add alert to history
    fn add_to_history(&self, alert: AlertEvent) -> SklResult<()> {
        let mut history = self.alert_history.lock().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire lock for alert history".to_string())
        })?;

        history.push_back(alert);

        // Maintain history size limit
        while history.len() > self.config.max_history {
            history.pop_front();
        }

        Ok(())
    }

    /// Update alert statistics
    fn update_stats(&self, alert: &AlertEvent) -> SklResult<()> {
        let mut stats = self.stats.write().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire write lock for stats".to_string())
        })?;

        stats.total_alerts += 1;
        *stats
            .alerts_by_severity
            .entry(alert.severity.clone())
            .or_insert(0) += 1;
        *stats
            .alerts_by_rule
            .entry(alert.rule_id.clone())
            .or_insert(0) += 1;

        let active_alerts = self.active_alerts.read().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire read lock for active alerts".to_string(),
            )
        })?;
        stats.active_alert_count = active_alerts.len();

        Ok(())
    }

    /// Update resolution statistics
    fn update_resolution_stats(&self, active_alert: &ActiveAlert) -> SklResult<()> {
        let mut stats = self.stats.write().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire write lock for stats".to_string())
        })?;

        let resolution_time = Utc::now() - active_alert.first_fired;
        let resolution_duration = Duration::from_millis(resolution_time.num_milliseconds() as u64);

        // Update average resolution time (simplified)
        stats.avg_resolution_time = resolution_duration;

        let active_alerts = self.active_alerts.read().map_err(|_| {
            SklearsError::InvalidOperation(
                "Failed to acquire read lock for active alerts".to_string(),
            )
        })?;
        stats.active_alert_count = active_alerts.len();

        Ok(())
    }
}

/// Implementation for console alert channel
impl ConsoleAlertChannel {
    /// Create new console alert channel
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            config: HashMap::new(),
        }
    }
}

impl AlertChannel for ConsoleAlertChannel {
    fn send_alert(&self, alert: &AlertEvent) -> SklResult<()> {
        println!(
            "🚨 ALERT [{}] {}: {}",
            alert.severity, alert.rule_id, alert.message
        );

        if let Some(pipeline) = &alert.pipeline {
            println!("   Pipeline: {pipeline}");
        }

        if let Some(stage) = &alert.stage {
            println!("   Stage: {stage}");
        }

        if let Some(value) = alert.trigger_value {
            println!("   Trigger Value: {value}");
        }

        println!("   Timestamp: {}", alert.timestamp);

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn health_check(&self) -> SklResult<()> {
        Ok(())
    }

    fn config(&self) -> HashMap<String, String> {
        self.config.clone()
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            max_history: 10000,
            default_cooldown: Duration::from_secs(300), // 5 minutes
            enable_grouping: true,
            grouping_window: Duration::from_secs(60), // 1 minute
            max_alerts_per_group: 10,
            enable_escalation: false,
            escalation_levels: Vec::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_automated_alerter_creation() {
        let config = AlertConfig::default();
        let alerter = AutomatedAlerter::new(config);

        let stats = alerter.get_stats().unwrap();
        assert_eq!(stats.total_alerts, 0);
        assert_eq!(stats.active_alert_count, 0);
    }

    #[test]
    fn test_alert_rule_management() {
        let config = AlertConfig::default();
        let alerter = AutomatedAlerter::new(config);

        let rule = AlertRule {
            id: "test_rule".to_string(),
            name: "Test Rule".to_string(),
            description: "Test alert rule".to_string(),
            condition: AlertCondition::Threshold {
                metric: "test_metric".to_string(),
                operator: ThresholdOperator::GreaterThan,
                value: 100.0,
                window: Duration::from_secs(60),
                min_points: 1,
            },
            severity: AlertSeverity::Warning,
            channels: vec!["console".to_string()],
            cooldown: Duration::from_secs(300),
            enabled: true,
            labels: HashMap::new(),
            priority: 1,
        };

        alerter.add_rule(rule).unwrap();

        let rules = alerter.rules.read().unwrap();
        assert!(rules.contains_key("test_rule"));
    }

    #[test]
    fn test_alert_channel_management() {
        let config = AlertConfig::default();
        let alerter = AutomatedAlerter::new(config);

        let channel = Box::new(ConsoleAlertChannel::new("test_console"));
        alerter.add_channel("console", channel).unwrap();

        let channels = alerter.channels.read().unwrap();
        assert!(channels.contains_key("console"));
    }

    #[test]
    fn test_console_alert_channel() {
        let channel = ConsoleAlertChannel::new("test");

        let alert = AlertEvent {
            id: "test_alert".to_string(),
            rule_id: "test_rule".to_string(),
            severity: AlertSeverity::Warning,
            message: "Test alert message".to_string(),
            timestamp: Utc::now(),
            pipeline: Some("test_pipeline".to_string()),
            stage: Some("test_stage".to_string()),
            labels: HashMap::new(),
            metadata: HashMap::new(),
            trigger_value: Some(150.0),
            status: AlertStatus::Firing,
        };

        assert!(channel.send_alert(&alert).is_ok());
        assert!(channel.health_check().is_ok());
        assert_eq!(channel.name(), "test");
    }

    #[test]
    fn test_threshold_condition_evaluation() {
        let config = AlertConfig::default();
        let alerter = AutomatedAlerter::new(config);

        let condition = AlertCondition::Threshold {
            metric: "cpu_usage".to_string(),
            operator: ThresholdOperator::GreaterThan,
            value: 80.0,
            window: Duration::from_secs(60),
            min_points: 1,
        };

        let metrics = vec![Metric {
            name: "cpu_usage".to_string(),
            value: 90.0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            pipeline_name: "test_pipeline".to_string(),
            stage_name: None,
            execution_id: None,
            metadata: HashMap::new(),
        }];

        assert!(alerter.evaluate_condition(&condition, &metrics).unwrap());
    }

    #[test]
    fn test_alert_silence() {
        let config = AlertConfig::default();
        let alerter = AutomatedAlerter::new(config);

        let silence_id = alerter
            .create_silence(
                vec!["test_rule".to_string()],
                Duration::from_secs(3600),
                "Maintenance window".to_string(),
                "admin".to_string(),
            )
            .unwrap();

        assert!(!silence_id.is_empty());
        assert!(alerter.is_silenced("test_rule").unwrap());

        alerter.remove_silence(&silence_id).unwrap();
        assert!(!alerter.is_silenced("test_rule").unwrap());
    }

    #[test]
    fn test_alert_acknowledgment() {
        let config = AlertConfig::default();
        let alerter = AutomatedAlerter::new(config);

        let alert = AlertEvent {
            id: "test_alert".to_string(),
            rule_id: "test_rule".to_string(),
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            timestamp: Utc::now(),
            pipeline: None,
            stage: None,
            labels: HashMap::new(),
            metadata: HashMap::new(),
            trigger_value: None,
            status: AlertStatus::Firing,
        };

        // Add as active alert
        let active_alert = ActiveAlert {
            event: alert.clone(),
            first_fired: alert.timestamp,
            last_updated: alert.timestamp,
            fire_count: 1,
            escalation_level: 0,
            channels_notified: Vec::new(),
        };

        {
            let mut active_alerts = alerter.active_alerts.write().unwrap();
            active_alerts.insert(alert.id.clone(), active_alert);
        }

        alerter.acknowledge_alert(&alert.id, "admin").unwrap();

        let active_alerts = alerter.active_alerts.read().unwrap();
        let acknowledged_alert = active_alerts.get(&alert.id).unwrap();
        assert_eq!(acknowledged_alert.event.status, AlertStatus::Acknowledged);
    }
}
