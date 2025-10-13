//! Alerting System and Notification Management
//!
//! This module provides comprehensive alerting capabilities including rule evaluation,
//! notification channels, alert suppression, escalation, and lifecycle management
//! for monitoring system alerts and notifications.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};

use crate::monitoring_core::{ActiveAlert, SeverityLevel, TimeRange};
use crate::metrics_collection::PerformanceMetric;
use crate::event_tracking::TaskExecutionEvent;
use crate::configuration_management::AlertConfig;

/// Alert manager for comprehensive alert handling
///
/// Manages alert rules, evaluates conditions, sends notifications, and handles
/// alert lifecycle including suppression and escalation.
#[derive(Debug)]
pub struct AlertManager {
    /// Active alert rules
    rules: HashMap<String, AlertRule>,

    /// Active alerts
    active_alerts: HashMap<String, ActiveAlert>,

    /// Alert state tracking
    alert_states: HashMap<String, AlertState>,

    /// Notification channels
    channels: Vec<Box<dyn AlertChannel>>,

    /// Suppression manager
    suppression_manager: AlertSuppressionManager,

    /// Alert statistics
    stats: AlertStatistics,

    /// Configuration
    config: AlertManagerConfig,

    /// Thread safety lock
    lock: Arc<RwLock<()>>,
}

impl AlertManager {
    /// Create new alert manager
    pub fn new(config: AlertManagerConfig) -> Self {
        Self {
            rules: HashMap::new(),
            active_alerts: HashMap::new(),
            alert_states: HashMap::new(),
            channels: Vec::new(),
            suppression_manager: AlertSuppressionManager::new(),
            stats: AlertStatistics::new(),
            config,
            lock: Arc::new(RwLock::new(())),
        }
    }

    /// Register alert rule
    pub fn register_rule(&mut self, rule: AlertRule) -> SklResult<()> {
        let _lock = self.lock.write().unwrap();

        // Validate rule
        rule.validate()?;

        // Check for duplicate rule names
        if self.rules.contains_key(&rule.name) {
            return Err(SklearsError::InvalidInput(
                format!("Alert rule '{}' already exists", rule.name)
            ));
        }

        self.rules.insert(rule.name.clone(), rule);
        Ok(())
    }

    /// Remove alert rule
    pub fn remove_rule(&mut self, rule_name: &str) -> SklResult<AlertRule> {
        let _lock = self.lock.write().unwrap();

        self.rules.remove(rule_name)
            .ok_or_else(|| SklearsError::NotFound(format!("Alert rule '{}' not found", rule_name)))
    }

    /// Register notification channel
    pub fn register_channel(&mut self, channel: Box<dyn AlertChannel>) {
        let _lock = self.lock.write().unwrap();
        self.channels.push(channel);
    }

    /// Evaluate all alert rules against metrics
    pub fn evaluate_rules(&mut self, metrics: &[PerformanceMetric]) -> SklResult<Vec<AlertAction>> {
        let _lock = self.lock.write().unwrap();
        let mut actions = Vec::new();

        for rule in self.rules.values() {
            match self.evaluate_rule(rule, metrics) {
                Ok(Some(action)) => actions.push(action),
                Ok(None) => {} // No action needed
                Err(e) => {
                    log::warn!("Rule evaluation failed for '{}': {}", rule.name, e);
                    self.stats.evaluation_failures += 1;
                }
            }
        }

        self.stats.evaluations += 1;
        self.stats.last_evaluation = SystemTime::now();

        Ok(actions)
    }

    /// Evaluate single alert rule
    fn evaluate_rule(&mut self, rule: &AlertRule, metrics: &[PerformanceMetric]) -> SklResult<Option<AlertAction>> {
        // Get current alert state
        let current_state = self.alert_states.get(&rule.name).cloned()
            .unwrap_or_else(|| AlertState::new(rule.name.clone()));

        // Check if alert is suppressed
        if self.suppression_manager.is_suppressed(&rule.name) {
            return Ok(None);
        }

        // Evaluate condition
        let condition_met = rule.condition.evaluate(metrics)?;

        let action = match (condition_met, current_state.is_triggered()) {
            (true, false) => {
                // Condition met, not currently triggered - fire alert
                let alert_id = self.generate_alert_id();
                let alert = ActiveAlert {
                    alert_id: alert_id.clone(),
                    rule_name: rule.name.clone(),
                    severity: rule.severity.clone(),
                    message: rule.generate_message(metrics),
                    start_time: SystemTime::now(),
                    acknowledged: false,
                };

                self.active_alerts.insert(alert_id.clone(), alert.clone());
                self.alert_states.insert(rule.name.clone(), AlertState::triggered(rule.name.clone()));
                self.stats.alerts_fired += 1;

                Some(AlertAction::Fire { alert, channels: rule.channels.clone() })
            }
            (false, true) => {
                // Condition not met, currently triggered - resolve alert
                if let Some(alert_id) = current_state.triggered_at {
                    if let Some(mut alert) = self.active_alerts.remove(&alert_id.to_string()) {
                        alert.acknowledged = true; // Mark as resolved
                        self.alert_states.insert(rule.name.clone(), AlertState::new(rule.name.clone()));
                        self.stats.alerts_resolved += 1;

                        Some(AlertAction::Resolve { alert, channels: rule.channels.clone() })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None, // No state change
        };

        Ok(action)
    }

    /// Send alert notifications
    pub fn send_notifications(&self, action: AlertAction) -> SklResult<()> {
        match action {
            AlertAction::Fire { alert, channels } => {
                for channel_name in &channels {
                    if let Some(channel) = self.find_channel(channel_name) {
                        if let Err(e) = channel.send_alert(&alert) {
                            log::error!("Failed to send alert to channel '{}': {}", channel_name, e);
                            // Continue with other channels
                        }
                    } else {
                        log::warn!("Alert channel '{}' not found", channel_name);
                    }
                }
            }
            AlertAction::Resolve { alert, channels } => {
                for channel_name in &channels {
                    if let Some(channel) = self.find_channel(channel_name) {
                        if let Err(e) = channel.send_resolution(&alert) {
                            log::error!("Failed to send resolution to channel '{}': {}", channel_name, e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Find notification channel by name
    fn find_channel(&self, name: &str) -> Option<&dyn AlertChannel> {
        self.channels.iter().find(|c| c.name() == name).map(|c| c.as_ref())
    }

    /// Generate unique alert ID
    fn generate_alert_id(&self) -> String {
        format!("alert_{}", uuid::Uuid::new_v4())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<ActiveAlert> {
        let _lock = self.lock.read().unwrap();
        self.active_alerts.values().cloned().collect()
    }

    /// Acknowledge alert
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> SklResult<()> {
        let _lock = self.lock.write().unwrap();

        if let Some(alert) = self.active_alerts.get_mut(alert_id) {
            alert.acknowledged = true;
            self.stats.alerts_acknowledged += 1;
            Ok(())
        } else {
            Err(SklearsError::NotFound(format!("Alert '{}' not found", alert_id)))
        }
    }

    /// Get alert statistics
    pub fn statistics(&self) -> &AlertStatistics {
        &self.stats
    }

    /// Configure alert suppression
    pub fn configure_suppression(&mut self, config: SuppressionConfig) -> SklResult<()> {
        self.suppression_manager.configure(config)
    }

    /// Cleanup old resolved alerts
    pub fn cleanup_resolved_alerts(&mut self, older_than: Duration) -> usize {
        let _lock = self.lock.write().unwrap();
        let cutoff_time = SystemTime::now() - older_than;
        let initial_count = self.active_alerts.len();

        self.active_alerts.retain(|_id, alert| {
            !alert.acknowledged || alert.start_time >= cutoff_time
        });

        initial_count - self.active_alerts.len()
    }
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name/identifier
    pub name: String,

    /// Alert condition
    pub condition: AlertCondition,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Alert message template
    pub message_template: String,

    /// Notification channels
    pub channels: Vec<String>,

    /// Rule enabled flag
    pub enabled: bool,

    /// Evaluation frequency
    pub evaluation_frequency: Duration,

    /// Rule metadata
    pub metadata: HashMap<String, String>,

    /// Rule tags
    pub tags: Vec<String>,

    /// Escalation configuration
    pub escalation: Option<EscalationConfig>,
}

impl AlertRule {
    /// Create new alert rule
    pub fn new(name: String, condition: AlertCondition, severity: SeverityLevel) -> Self {
        Self {
            name,
            condition,
            severity,
            message_template: "Alert triggered".to_string(),
            channels: Vec::new(),
            enabled: true,
            evaluation_frequency: Duration::from_secs(60),
            metadata: HashMap::new(),
            tags: Vec::new(),
            escalation: None,
        }
    }

    /// Builder pattern for rule creation
    pub fn builder() -> AlertRuleBuilder {
        AlertRuleBuilder::new()
    }

    /// Validate alert rule
    pub fn validate(&self) -> SklResult<()> {
        if self.name.is_empty() {
            return Err(SklearsError::InvalidInput("Alert rule name cannot be empty".to_string()));
        }

        if self.channels.is_empty() {
            return Err(SklearsError::InvalidInput("Alert rule must have at least one channel".to_string()));
        }

        self.condition.validate()?;

        Ok(())
    }

    /// Generate alert message from template and metrics
    pub fn generate_message(&self, metrics: &[PerformanceMetric]) -> String {
        // Simple template replacement - could be enhanced with proper templating
        let mut message = self.message_template.clone();

        // Replace common placeholders
        if let Some(metric) = metrics.first() {
            message = message.replace("{metric_name}", &metric.name);
            message = message.replace("{metric_value}", &metric.value.to_string());
            message = message.replace("{metric_unit}", &metric.unit);
        }

        message = message.replace("{rule_name}", &self.name);
        message = message.replace("{severity}", &format!("{:?}", self.severity));
        message = message.replace("{timestamp}", &format!("{:?}", SystemTime::now()));

        message
    }
}

/// Alert rule builder
#[derive(Debug)]
pub struct AlertRuleBuilder {
    name: Option<String>,
    condition: Option<AlertCondition>,
    severity: SeverityLevel,
    message_template: String,
    channels: Vec<String>,
    enabled: bool,
    evaluation_frequency: Duration,
    metadata: HashMap<String, String>,
    tags: Vec<String>,
    escalation: Option<EscalationConfig>,
}

impl AlertRuleBuilder {
    pub fn new() -> Self {
        Self {
            name: None,
            condition: None,
            severity: SeverityLevel::Medium,
            message_template: "Alert triggered".to_string(),
            channels: Vec::new(),
            enabled: true,
            evaluation_frequency: Duration::from_secs(60),
            metadata: HashMap::new(),
            tags: Vec::new(),
            escalation: None,
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn condition(mut self, condition: AlertCondition) -> Self {
        self.condition = Some(condition);
        self
    }

    pub fn severity(mut self, severity: SeverityLevel) -> Self {
        self.severity = severity;
        self
    }

    pub fn message_template(mut self, template: String) -> Self {
        self.message_template = template;
        self
    }

    pub fn channel(mut self, channel: String) -> Self {
        self.channels.push(channel);
        self
    }

    pub fn build(self) -> SklResult<AlertRule> {
        let name = self.name.ok_or_else(|| SklearsError::InvalidInput("Rule name is required".to_string()))?;
        let condition = self.condition.ok_or_else(|| SklearsError::InvalidInput("Rule condition is required".to_string()))?;

        Ok(AlertRule {
            name,
            condition,
            severity: self.severity,
            message_template: self.message_template,
            channels: self.channels,
            enabled: self.enabled,
            evaluation_frequency: self.evaluation_frequency,
            metadata: self.metadata,
            tags: self.tags,
            escalation: self.escalation,
        })
    }
}

impl Default for AlertRuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert conditions for rule evaluation
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold-based condition
    Threshold {
        metric: String,
        operator: ComparisonOperator,
        value: f64,
        duration: Duration,
    },

    /// Rate-based condition
    Rate {
        metric: String,
        rate_threshold: f64,
        window: Duration,
    },

    /// Anomaly detection condition
    Anomaly {
        metric: String,
        sensitivity: f64,
        window: Duration,
    },

    /// Composite condition (AND/OR)
    Composite {
        operator: LogicalOperator,
        conditions: Vec<AlertCondition>,
    },

    /// Custom condition with expression
    Custom {
        expression: String,
        variables: HashMap<String, String>,
    },
}

impl AlertCondition {
    /// Validate alert condition
    pub fn validate(&self) -> SklResult<()> {
        match self {
            AlertCondition::Threshold { metric, duration, .. } => {
                if metric.is_empty() {
                    return Err(SklearsError::InvalidInput("Metric name cannot be empty".to_string()));
                }
                if duration.as_secs() == 0 {
                    return Err(SklearsError::InvalidInput("Duration must be greater than zero".to_string()));
                }
            }
            AlertCondition::Rate { metric, window, .. } => {
                if metric.is_empty() {
                    return Err(SklearsError::InvalidInput("Metric name cannot be empty".to_string()));
                }
                if window.as_secs() == 0 {
                    return Err(SklearsError::InvalidInput("Window must be greater than zero".to_string()));
                }
            }
            AlertCondition::Composite { conditions, .. } => {
                if conditions.is_empty() {
                    return Err(SklearsError::InvalidInput("Composite condition must have at least one sub-condition".to_string()));
                }
                for condition in conditions {
                    condition.validate()?;
                }
            }
            _ => {} // Other conditions are assumed valid for now
        }

        Ok(())
    }

    /// Evaluate condition against metrics
    pub fn evaluate(&self, metrics: &[PerformanceMetric]) -> SklResult<bool> {
        match self {
            AlertCondition::Threshold { metric, operator, value, duration: _ } => {
                // Find matching metrics
                let matching_metrics: Vec<&PerformanceMetric> = metrics.iter()
                    .filter(|m| m.name == *metric)
                    .collect();

                if matching_metrics.is_empty() {
                    return Ok(false);
                }

                // For simplicity, use the latest metric value
                let latest_metric = matching_metrics.iter()
                    .max_by_key(|m| m.timestamp)
                    .unwrap();

                Ok(operator.compare(latest_metric.value, *value))
            }
            AlertCondition::Rate { metric, rate_threshold, window } => {
                // Calculate rate over window
                let now = SystemTime::now();
                let window_start = now - *window;

                let matching_metrics: Vec<&PerformanceMetric> = metrics.iter()
                    .filter(|m| m.name == *metric && m.timestamp >= window_start)
                    .collect();

                if matching_metrics.len() < 2 {
                    return Ok(false);
                }

                let rate = matching_metrics.len() as f64 / window.as_secs_f64();
                Ok(rate > *rate_threshold)
            }
            AlertCondition::Composite { operator, conditions } => {
                let results: Result<Vec<bool>, SklearsError> = conditions.iter()
                    .map(|c| c.evaluate(metrics))
                    .collect();

                let results = results?;

                Ok(match operator {
                    LogicalOperator::And => results.iter().all(|&x| x),
                    LogicalOperator::Or => results.iter().any(|&x| x),
                    LogicalOperator::Not => {
                        if results.len() != 1 {
                            return Err(SklearsError::InvalidInput("NOT operator requires exactly one condition".to_string()));
                        }
                        !results[0]
                    }
                })
            }
            _ => {
                // Other condition types not implemented yet
                Ok(false)
            }
        }
    }
}

/// Comparison operators for threshold conditions
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Equal,
    NotEqual,
}

impl ComparisonOperator {
    /// Compare values using this operator
    pub fn compare(&self, left: f64, right: f64) -> bool {
        match self {
            ComparisonOperator::Greater => left > right,
            ComparisonOperator::GreaterOrEqual => left >= right,
            ComparisonOperator::Less => left < right,
            ComparisonOperator::LessOrEqual => left <= right,
            ComparisonOperator::Equal => (left - right).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (left - right).abs() >= f64::EPSILON,
        }
    }
}

/// Logical operators for composite conditions
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

/// Alert actions generated by rule evaluation
#[derive(Debug, Clone)]
pub enum AlertAction {
    /// Fire new alert
    Fire {
        alert: ActiveAlert,
        channels: Vec<String>,
    },

    /// Resolve existing alert
    Resolve {
        alert: ActiveAlert,
        channels: Vec<String>,
    },
}

/// Alert state tracking
#[derive(Debug, Clone)]
pub struct AlertState {
    /// Rule identifier
    pub rule_id: String,

    /// When alert was triggered (if active)
    pub triggered_at: Option<SystemTime>,

    /// Last evaluation timestamp
    pub last_evaluation: SystemTime,

    /// Number of evaluations
    pub evaluation_count: usize,

    /// Consecutive trigger count
    pub consecutive_triggers: usize,
}

impl AlertState {
    /// Create new alert state
    pub fn new(rule_id: String) -> Self {
        Self {
            rule_id,
            triggered_at: None,
            last_evaluation: SystemTime::now(),
            evaluation_count: 0,
            consecutive_triggers: 0,
        }
    }

    /// Create triggered alert state
    pub fn triggered(rule_id: String) -> Self {
        Self {
            rule_id,
            triggered_at: Some(SystemTime::now()),
            last_evaluation: SystemTime::now(),
            evaluation_count: 1,
            consecutive_triggers: 1,
        }
    }

    /// Check if alert is currently triggered
    pub fn is_triggered(&self) -> bool {
        self.triggered_at.is_some()
    }

    /// Get alert duration if triggered
    pub fn duration(&self) -> Option<Duration> {
        self.triggered_at.map(|start| {
            SystemTime::now().duration_since(start).unwrap_or(Duration::from_secs(0))
        })
    }
}

/// Alert notification channel trait
pub trait AlertChannel: Send + Sync {
    /// Send alert notification
    fn send_alert(&self, alert: &ActiveAlert) -> SklResult<()>;

    /// Send alert resolution notification
    fn send_resolution(&self, alert: &ActiveAlert) -> SklResult<()>;

    /// Get channel name
    fn name(&self) -> &str;

    /// Check if channel is available
    fn is_available(&self) -> bool;

    /// Get channel configuration
    fn config(&self) -> ChannelConfig;
}

/// Channel configuration
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Channel name
    pub name: String,

    /// Channel type
    pub channel_type: ChannelType,

    /// Channel-specific settings
    pub settings: HashMap<String, String>,

    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,

    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Types of notification channels
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelType {
    Email,
    Slack,
    Webhook,
    SMS,
    PagerDuty,
    Console,
    Log,
    Custom { type_name: String },
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum notifications per time window
    pub max_per_window: usize,

    /// Time window duration
    pub window_duration: Duration,

    /// Enable rate limiting
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_per_window: 10,
            window_duration: Duration::from_secs(60),
            enabled: true,
        }
    }
}

/// Retry configuration for failed notifications
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,

    /// Initial retry delay
    pub initial_delay: Duration,

    /// Backoff multiplier
    pub backoff_multiplier: f64,

    /// Maximum retry delay
    pub max_delay: Duration,

    /// Enable retries
    pub enabled: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(300),
            enabled: true,
        }
    }
}

/// Alert suppression manager
#[derive(Debug)]
pub struct AlertSuppressionManager {
    /// Suppression rules
    rules: Vec<SuppressionRule>,

    /// Active suppressions
    active_suppressions: HashMap<String, SuppressionState>,

    /// Configuration
    config: SuppressionConfig,
}

impl AlertSuppressionManager {
    /// Create new suppression manager
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            active_suppressions: HashMap::new(),
            config: SuppressionConfig::default(),
        }
    }

    /// Configure suppression
    pub fn configure(&mut self, config: SuppressionConfig) -> SklResult<()> {
        self.config = config;
        Ok(())
    }

    /// Check if alert is suppressed
    pub fn is_suppressed(&self, rule_name: &str) -> bool {
        // Check active suppressions
        if let Some(suppression) = self.active_suppressions.get(rule_name) {
            if suppression.is_active() {
                return true;
            }
        }

        // Check suppression rules
        for rule in &self.rules {
            if rule.matches(rule_name) {
                return true;
            }
        }

        false
    }

    /// Add suppression rule
    pub fn add_suppression_rule(&mut self, rule: SuppressionRule) {
        self.rules.push(rule);
    }

    /// Remove suppression rule
    pub fn remove_suppression_rule(&mut self, rule_name: &str) {
        self.rules.retain(|rule| rule.name != rule_name);
    }
}

impl Default for AlertSuppressionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Suppression rule
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,

    /// Pattern to match alert names
    pub pattern: String,

    /// Suppression duration
    pub duration: Duration,

    /// Rule enabled flag
    pub enabled: bool,
}

impl SuppressionRule {
    /// Check if rule matches alert name
    pub fn matches(&self, alert_name: &str) -> bool {
        if !self.enabled {
            return false;
        }

        // Simple pattern matching - could be enhanced with regex
        alert_name.contains(&self.pattern)
    }
}

/// Suppression state
#[derive(Debug, Clone)]
pub struct SuppressionState {
    /// Start time
    pub start_time: SystemTime,

    /// Duration
    pub duration: Duration,

    /// Reason for suppression
    pub reason: String,
}

impl SuppressionState {
    /// Check if suppression is currently active
    pub fn is_active(&self) -> bool {
        let elapsed = SystemTime::now().duration_since(self.start_time).unwrap_or(Duration::from_secs(0));
        elapsed < self.duration
    }
}

/// Suppression configuration
#[derive(Debug, Clone)]
pub struct SuppressionConfig {
    /// Enable suppression
    pub enabled: bool,

    /// Default suppression duration
    pub default_duration: Duration,

    /// Maximum suppression duration
    pub max_duration: Duration,

    /// Auto-cleanup expired suppressions
    pub auto_cleanup: bool,
}

impl Default for SuppressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_duration: Duration::from_secs(3600), // 1 hour
            max_duration: Duration::from_secs(24 * 3600), // 24 hours
            auto_cleanup: true,
        }
    }
}

/// Escalation configuration
#[derive(Debug, Clone)]
pub struct EscalationConfig {
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,

    /// Enable escalation
    pub enabled: bool,
}

/// Escalation level
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Time before escalation
    pub delay: Duration,

    /// Target channels for this level
    pub channels: Vec<String>,

    /// Escalation message template
    pub message_template: String,
}

/// Alert manager configuration
#[derive(Debug, Clone)]
pub struct AlertManagerConfig {
    /// Maximum number of active alerts
    pub max_active_alerts: usize,

    /// Alert cleanup interval
    pub cleanup_interval: Duration,

    /// Enable alert deduplication
    pub enable_deduplication: bool,

    /// Deduplication window
    pub deduplication_window: Duration,

    /// Default evaluation frequency
    pub default_evaluation_frequency: Duration,
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            max_active_alerts: 1000,
            cleanup_interval: Duration::from_secs(3600),
            enable_deduplication: true,
            deduplication_window: Duration::from_secs(300),
            default_evaluation_frequency: Duration::from_secs(60),
        }
    }
}

/// Alert statistics
#[derive(Debug, Clone)]
pub struct AlertStatistics {
    /// Number of alerts fired
    pub alerts_fired: u64,

    /// Number of alerts resolved
    pub alerts_resolved: u64,

    /// Number of alerts acknowledged
    pub alerts_acknowledged: u64,

    /// Number of rule evaluations
    pub evaluations: u64,

    /// Number of evaluation failures
    pub evaluation_failures: u64,

    /// Last evaluation timestamp
    pub last_evaluation: SystemTime,

    /// Average evaluation time
    pub avg_evaluation_time: Duration,
}

impl AlertStatistics {
    fn new() -> Self {
        Self {
            alerts_fired: 0,
            alerts_resolved: 0,
            alerts_acknowledged: 0,
            evaluations: 0,
            evaluation_failures: 0,
            last_evaluation: SystemTime::now(),
            avg_evaluation_time: Duration::from_millis(0),
        }
    }

    /// Calculate alert resolution rate
    pub fn resolution_rate(&self) -> f64 {
        if self.alerts_fired > 0 {
            self.alerts_resolved as f64 / self.alerts_fired as f64
        } else {
            0.0
        }
    }

    /// Calculate evaluation success rate
    pub fn evaluation_success_rate(&self) -> f64 {
        let total = self.evaluations + self.evaluation_failures;
        if total > 0 {
            self.evaluations as f64 / total as f64
        } else {
            1.0
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_rule_creation() {
        let condition = AlertCondition::Threshold {
            metric: "cpu_usage".to_string(),
            operator: ComparisonOperator::Greater,
            value: 0.8,
            duration: Duration::from_secs(60),
        };

        let rule = AlertRule::new(
            "high_cpu".to_string(),
            condition,
            SeverityLevel::High,
        );

        assert_eq!(rule.name, "high_cpu");
        assert_eq!(rule.severity, SeverityLevel::High);
        assert!(rule.enabled);
    }

    #[test]
    fn test_alert_rule_builder() {
        let rule = AlertRule::builder()
            .name("memory_alert".to_string())
            .condition(AlertCondition::Threshold {
                metric: "memory_usage".to_string(),
                operator: ComparisonOperator::Greater,
                value: 0.9,
                duration: Duration::from_secs(120),
            })
            .severity(SeverityLevel::Critical)
            .channel("email".to_string())
            .channel("slack".to_string())
            .build()
            .unwrap();

        assert_eq!(rule.name, "memory_alert");
        assert_eq!(rule.channels.len(), 2);
        assert!(rule.channels.contains(&"email".to_string()));
    }

    #[test]
    fn test_comparison_operators() {
        assert!(ComparisonOperator::Greater.compare(5.0, 3.0));
        assert!(!ComparisonOperator::Greater.compare(3.0, 5.0));

        assert!(ComparisonOperator::Less.compare(3.0, 5.0));
        assert!(!ComparisonOperator::Less.compare(5.0, 3.0));

        assert!(ComparisonOperator::Equal.compare(5.0, 5.0));
        assert!(!ComparisonOperator::Equal.compare(5.0, 3.0));
    }

    #[test]
    fn test_alert_condition_evaluation() {
        let metrics = vec![
            PerformanceMetric::new("cpu_usage".to_string(), 0.85, "percentage".to_string()),
            PerformanceMetric::new("memory_usage".to_string(), 0.75, "percentage".to_string()),
        ];

        let condition = AlertCondition::Threshold {
            metric: "cpu_usage".to_string(),
            operator: ComparisonOperator::Greater,
            value: 0.8,
            duration: Duration::from_secs(60),
        };

        let result = condition.evaluate(&metrics).unwrap();
        assert!(result); // 0.85 > 0.8

        let condition2 = AlertCondition::Threshold {
            metric: "cpu_usage".to_string(),
            operator: ComparisonOperator::Less,
            value: 0.8,
            duration: Duration::from_secs(60),
        };

        let result2 = condition2.evaluate(&metrics).unwrap();
        assert!(!result2); // 0.85 is not < 0.8
    }

    #[test]
    fn test_alert_state() {
        let state = AlertState::new("test_rule".to_string());
        assert!(!state.is_triggered());
        assert_eq!(state.duration(), None);

        let triggered_state = AlertState::triggered("test_rule".to_string());
        assert!(triggered_state.is_triggered());
        assert!(triggered_state.duration().is_some());
    }

    #[test]
    fn test_suppression_rule() {
        let rule = SuppressionRule {
            name: "suppress_cpu".to_string(),
            pattern: "cpu".to_string(),
            duration: Duration::from_secs(3600),
            enabled: true,
        };

        assert!(rule.matches("high_cpu_alert"));
        assert!(rule.matches("cpu_usage_warning"));
        assert!(!rule.matches("memory_alert"));

        let disabled_rule = SuppressionRule {
            enabled: false,
            ..rule
        };

        assert!(!disabled_rule.matches("high_cpu_alert"));
    }

    #[test]
    fn test_alert_manager() {
        let config = AlertManagerConfig::default();
        let mut manager = AlertManager::new(config);

        let rule = AlertRule::new(
            "test_rule".to_string(),
            AlertCondition::Threshold {
                metric: "test_metric".to_string(),
                operator: ComparisonOperator::Greater,
                value: 0.5,
                duration: Duration::from_secs(60),
            },
            SeverityLevel::Medium,
        );

        assert!(manager.register_rule(rule).is_ok());
        assert_eq!(manager.rules.len(), 1);

        // Test duplicate rule registration
        let duplicate_rule = AlertRule::new(
            "test_rule".to_string(),
            AlertCondition::Threshold {
                metric: "test_metric".to_string(),
                operator: ComparisonOperator::Greater,
                value: 0.5,
                duration: Duration::from_secs(60),
            },
            SeverityLevel::Medium,
        );

        assert!(manager.register_rule(duplicate_rule).is_err());
    }
}