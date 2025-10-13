//! Alert Systems and Notification Channels
//!
//! This module provides comprehensive alerting capabilities for the execution monitoring
//! framework. It includes alert rules, condition evaluation, multi-channel notifications,
//! escalation policies, alert suppression, and alert analytics.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use sklears_core::error::{Result as SklResult, SklearsError};
use crate::monitoring_config::*;
use crate::monitoring_metrics::*;
use crate::monitoring_events::*;
use crate::monitoring_core::*;

/// Alert manager for comprehensive alert handling
///
/// Coordinates alert evaluation, notification, escalation, and suppression
/// across multiple channels and policies.
#[derive(Debug)]
pub struct AlertManager {
    /// Alert rules
    rules: Vec<AlertRule>,

    /// Notification channels
    channels: HashMap<String, Box<dyn NotificationChannel>>,

    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,

    /// Alert history
    alert_history: Arc<RwLock<VecDeque<AlertHistoryEntry>>>,

    /// Suppression manager
    suppression_manager: AlertSuppressionManager,

    /// Escalation manager
    escalation_manager: AlertEscalationManager,

    /// Configuration
    config: AlertConfig,

    /// Manager statistics
    stats: Arc<RwLock<AlertManagerStats>>,
}

/// Alert rule for condition-based alerting
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule identifier
    pub id: String,

    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Alert condition
    pub condition: AlertCondition,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Target notification channels
    pub channels: Vec<String>,

    /// Rule configuration
    pub config: AlertRuleConfig,

    /// Rule state
    pub state: AlertRuleState,

    /// Last evaluation time
    pub last_evaluation: Option<SystemTime>,

    /// Evaluation count
    pub evaluation_count: u64,
}

/// Alert rule configuration
#[derive(Debug, Clone)]
pub struct AlertRuleConfig {
    /// Rule enabled status
    pub enabled: bool,

    /// Evaluation interval
    pub evaluation_interval: Duration,

    /// Evaluation timeout
    pub evaluation_timeout: Duration,

    /// Required consecutive triggers
    pub consecutive_triggers: usize,

    /// Recovery evaluation
    pub recovery_evaluation: bool,

    /// Auto-resolve after duration
    pub auto_resolve_after: Option<Duration>,

    /// Tags for rule categorization
    pub tags: HashMap<String, String>,
}

/// Alert rule state
#[derive(Debug, Clone)]
pub enum AlertRuleState {
    /// Rule is not triggered
    Normal,
    /// Rule is pending (not enough consecutive triggers)
    Pending { trigger_count: usize },
    /// Rule is triggered and alert is active
    Triggered,
    /// Rule is in recovery phase
    Recovery,
    /// Rule is disabled
    Disabled,
}

/// Alert conditions for triggering alerts
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold-based condition
    Threshold {
        metric_name: String,
        operator: ComparisonOperator,
        threshold: f64,
        window: Duration,
        aggregation: AggregationFunction,
    },

    /// Rate-based condition
    Rate {
        metric_name: String,
        rate_threshold: f64,
        window: Duration,
        direction: RateDirection,
    },

    /// Anomaly-based condition
    Anomaly {
        metric_name: String,
        algorithm: AnomalyDetectionAlgorithm,
        sensitivity: f64,
        baseline_window: Duration,
    },

    /// Event-based condition
    Event {
        event_type: TaskEventType,
        severity: SeverityLevel,
        count_threshold: usize,
        window: Duration,
    },

    /// Composite condition
    Composite {
        operator: LogicalOperator,
        conditions: Vec<AlertCondition>,
    },

    /// Expression-based condition
    Expression {
        expression: String,
        variables: HashMap<String, String>,
    },

    /// Custom condition
    Custom {
        evaluator: String,
        parameters: HashMap<String, String>,
    },
}

/// Rate directions for rate-based conditions
#[derive(Debug, Clone)]
pub enum RateDirection {
    Increasing,
    Decreasing,
    Both,
}

/// Comparison operators for threshold conditions
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual,
    Between { min: f64, max: f64 },
    NotBetween { min: f64, max: f64 },
}

/// Active alert information
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// Alert identifier
    pub alert_id: String,

    /// Rule that triggered the alert
    pub rule_id: String,

    /// Rule name
    pub rule_name: String,

    /// Alert message
    pub message: String,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Alert start time
    pub started_at: SystemTime,

    /// Last updated time
    pub updated_at: SystemTime,

    /// Alert state
    pub state: AlertState,

    /// Trigger value
    pub trigger_value: Option<f64>,

    /// Alert context
    pub context: AlertContext,

    /// Notification status
    pub notifications: Vec<NotificationStatus>,

    /// Escalation level
    pub escalation_level: usize,

    /// Acknowledgment status
    pub acknowledged: bool,

    /// Acknowledged by
    pub acknowledged_by: Option<String>,

    /// Alert tags
    pub tags: HashMap<String, String>,
}

/// Alert states
#[derive(Debug, Clone)]
pub enum AlertState {
    /// Alert is active and firing
    Firing,
    /// Alert is acknowledged but still active
    Acknowledged,
    /// Alert is resolved
    Resolved,
    /// Alert is suppressed
    Suppressed,
    /// Alert is escalated
    Escalated,
}

/// Alert context information
#[derive(Debug, Clone)]
pub struct AlertContext {
    /// Session ID where alert occurred
    pub session_id: String,

    /// Metric values at alert time
    pub metric_values: HashMap<String, f64>,

    /// Related events
    pub related_events: Vec<String>,

    /// Additional context data
    pub context_data: HashMap<String, String>,
}

/// Notification status
#[derive(Debug, Clone)]
pub struct NotificationStatus {
    /// Channel name
    pub channel: String,

    /// Notification time
    pub sent_at: SystemTime,

    /// Delivery status
    pub status: DeliveryStatus,

    /// Error message if failed
    pub error_message: Option<String>,

    /// Retry count
    pub retry_count: usize,
}

/// Delivery status for notifications
#[derive(Debug, Clone)]
pub enum DeliveryStatus {
    Pending,
    Sent,
    Delivered,
    Failed,
    Retrying,
}

/// Alert history entry
#[derive(Debug, Clone)]
pub struct AlertHistoryEntry {
    /// Alert ID
    pub alert_id: String,

    /// Rule ID
    pub rule_id: String,

    /// Action performed
    pub action: AlertAction,

    /// Timestamp
    pub timestamp: SystemTime,

    /// Performed by
    pub performed_by: Option<String>,

    /// Additional details
    pub details: HashMap<String, String>,
}

/// Alert actions for history tracking
#[derive(Debug, Clone)]
pub enum AlertAction {
    Triggered,
    Acknowledged,
    Resolved,
    Escalated,
    Suppressed,
    Notification { channel: String, status: DeliveryStatus },
}

/// Notification channel trait
pub trait NotificationChannel: Send + Sync {
    /// Send notification
    fn send_notification(&mut self, alert: &ActiveAlert) -> SklResult<NotificationResult>;

    /// Check channel health
    fn health_check(&self) -> SklResult<ChannelHealth>;

    /// Get channel name
    fn name(&self) -> &str;

    /// Get supported alert severities
    fn supported_severities(&self) -> Vec<SeverityLevel>;

    /// Check if channel supports batching
    fn supports_batching(&self) -> bool;

    /// Send batch notification
    fn send_batch_notification(&mut self, alerts: &[ActiveAlert]) -> SklResult<Vec<NotificationResult>> {
        // Default implementation sends individual notifications
        alerts.iter().map(|alert| self.send_notification(alert)).collect()
    }
}

/// Notification result
#[derive(Debug, Clone)]
pub struct NotificationResult {
    /// Delivery status
    pub status: DeliveryStatus,

    /// Message ID (if applicable)
    pub message_id: Option<String>,

    /// Error message
    pub error_message: Option<String>,

    /// Delivery timestamp
    pub delivered_at: Option<SystemTime>,
}

/// Channel health information
#[derive(Debug, Clone)]
pub struct ChannelHealth {
    /// Channel status
    pub status: ChannelStatus,

    /// Health score (0.0 to 1.0)
    pub score: f64,

    /// Last successful notification
    pub last_success: Option<SystemTime>,

    /// Recent failure count
    pub recent_failures: usize,

    /// Average delivery time
    pub avg_delivery_time: Duration,
}

/// Channel status
#[derive(Debug, Clone)]
pub enum ChannelStatus {
    Healthy,
    Degraded,
    Unavailable,
    Unknown,
}

/// Email notification channel
#[derive(Debug)]
pub struct EmailChannel {
    /// Channel configuration
    config: EmailChannelConfig,

    /// Channel statistics
    stats: Arc<RwLock<ChannelStats>>,
}

/// Email channel configuration
#[derive(Debug, Clone)]
pub struct EmailChannelConfig {
    /// SMTP server
    pub smtp_server: String,

    /// SMTP port
    pub smtp_port: u16,

    /// Username
    pub username: String,

    /// Password
    pub password: String,

    /// From email address
    pub from_email: String,

    /// Default recipients
    pub default_recipients: Vec<String>,

    /// Use TLS
    pub use_tls: bool,

    /// Connection timeout
    pub timeout: Duration,

    /// Email template
    pub template: EmailTemplate,
}

/// Email template
#[derive(Debug, Clone)]
pub struct EmailTemplate {
    /// Subject template
    pub subject: String,

    /// Body template (HTML)
    pub body_html: String,

    /// Body template (plain text)
    pub body_text: String,

    /// Include alert details
    pub include_details: bool,
}

/// Channel statistics
#[derive(Debug, Clone)]
pub struct ChannelStats {
    /// Total notifications sent
    pub total_sent: u64,

    /// Successful deliveries
    pub successful_deliveries: u64,

    /// Failed deliveries
    pub failed_deliveries: u64,

    /// Average delivery time
    pub avg_delivery_time: Duration,

    /// Last activity
    pub last_activity: Option<SystemTime>,
}

impl EmailChannel {
    /// Create new email channel
    pub fn new(config: EmailChannelConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ChannelStats::default())),
        }
    }

    /// Format email content
    fn format_email(&self, alert: &ActiveAlert) -> (String, String, String) {
        let subject = self.config.template.subject
            .replace("{alert_name}", &alert.rule_name)
            .replace("{severity}", &format!("{:?}", alert.severity))
            .replace("{alert_id}", &alert.alert_id);

        let body_html = self.config.template.body_html
            .replace("{alert_name}", &alert.rule_name)
            .replace("{message}", &alert.message)
            .replace("{severity}", &format!("{:?}", alert.severity))
            .replace("{started_at}", &format!("{:?}", alert.started_at))
            .replace("{alert_id}", &alert.alert_id);

        let body_text = self.config.template.body_text
            .replace("{alert_name}", &alert.rule_name)
            .replace("{message}", &alert.message)
            .replace("{severity}", &format!("{:?}", alert.severity))
            .replace("{started_at}", &format!("{:?}", alert.started_at))
            .replace("{alert_id}", &alert.alert_id);

        (subject, body_html, body_text)
    }
}

impl NotificationChannel for EmailChannel {
    fn send_notification(&mut self, alert: &ActiveAlert) -> SklResult<NotificationResult> {
        let start_time = SystemTime::now();
        let mut stats = self.stats.write().unwrap();

        // Format email content
        let (subject, body_html, body_text) = self.format_email(alert);

        // Simulate email sending (in real implementation would use SMTP library)
        let result = if self.config.smtp_server.is_empty() {
            NotificationResult {
                status: DeliveryStatus::Failed,
                message_id: None,
                error_message: Some("SMTP server not configured".to_string()),
                delivered_at: None,
            }
        } else {
            // Simulate successful delivery
            NotificationResult {
                status: DeliveryStatus::Delivered,
                message_id: Some(format!("email_{}", uuid::Uuid::new_v4())),
                error_message: None,
                delivered_at: Some(SystemTime::now()),
            }
        };

        // Update statistics
        stats.total_sent += 1;
        match result.status {
            DeliveryStatus::Delivered => stats.successful_deliveries += 1,
            DeliveryStatus::Failed => stats.failed_deliveries += 1,
            _ => {}
        }

        if let Ok(elapsed) = start_time.elapsed() {
            stats.avg_delivery_time = (stats.avg_delivery_time * (stats.total_sent - 1) as u32 + elapsed) / stats.total_sent as u32;
        }

        stats.last_activity = Some(SystemTime::now());

        Ok(result)
    }

    fn health_check(&self) -> SklResult<ChannelHealth> {
        let stats = self.stats.read().unwrap();

        let score = if stats.total_sent == 0 {
            1.0 // No data, assume healthy
        } else {
            stats.successful_deliveries as f64 / stats.total_sent as f64
        };

        let status = if score >= 0.95 {
            ChannelStatus::Healthy
        } else if score >= 0.8 {
            ChannelStatus::Degraded
        } else {
            ChannelStatus::Unavailable
        };

        Ok(ChannelHealth {
            status,
            score,
            last_success: stats.last_activity,
            recent_failures: stats.failed_deliveries as usize,
            avg_delivery_time: stats.avg_delivery_time,
        })
    }

    fn name(&self) -> &str {
        "email"
    }

    fn supported_severities(&self) -> Vec<SeverityLevel> {
        vec![SeverityLevel::Low, SeverityLevel::Medium, SeverityLevel::High, SeverityLevel::Critical]
    }

    fn supports_batching(&self) -> bool {
        true
    }
}

/// Slack notification channel
#[derive(Debug)]
pub struct SlackChannel {
    /// Channel configuration
    config: SlackChannelConfig,

    /// Channel statistics
    stats: Arc<RwLock<ChannelStats>>,
}

/// Slack channel configuration
#[derive(Debug, Clone)]
pub struct SlackChannelConfig {
    /// Webhook URL
    pub webhook_url: String,

    /// Default channel
    pub channel: String,

    /// Bot username
    pub username: String,

    /// Icon emoji
    pub icon_emoji: String,

    /// Message template
    pub template: SlackTemplate,

    /// Timeout
    pub timeout: Duration,
}

/// Slack message template
#[derive(Debug, Clone)]
pub struct SlackTemplate {
    /// Message text template
    pub text: String,

    /// Use rich formatting
    pub rich_formatting: bool,

    /// Include attachments
    pub include_attachments: bool,
}

impl SlackChannel {
    /// Create new Slack channel
    pub fn new(config: SlackChannelConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ChannelStats::default())),
        }
    }

    /// Format Slack message
    fn format_message(&self, alert: &ActiveAlert) -> String {
        let emoji = match alert.severity {
            SeverityLevel::Critical => "🚨",
            SeverityLevel::High => "⚠️",
            SeverityLevel::Medium => "⚡",
            SeverityLevel::Low => "ℹ️",
            _ => "📊",
        };

        format!(
            "{} *{}* - {} ({})\nStarted: {:?}\nAlert ID: {}",
            emoji,
            alert.rule_name,
            alert.message,
            format!("{:?}", alert.severity),
            alert.started_at,
            alert.alert_id
        )
    }
}

impl NotificationChannel for SlackChannel {
    fn send_notification(&mut self, alert: &ActiveAlert) -> SklResult<NotificationResult> {
        let start_time = SystemTime::now();
        let mut stats = self.stats.write().unwrap();

        // Format message
        let message = self.format_message(alert);

        // Simulate Slack webhook call (in real implementation would use HTTP client)
        let result = if self.config.webhook_url.is_empty() {
            NotificationResult {
                status: DeliveryStatus::Failed,
                message_id: None,
                error_message: Some("Webhook URL not configured".to_string()),
                delivered_at: None,
            }
        } else {
            // Simulate successful delivery
            NotificationResult {
                status: DeliveryStatus::Delivered,
                message_id: Some(format!("slack_{}", uuid::Uuid::new_v4())),
                error_message: None,
                delivered_at: Some(SystemTime::now()),
            }
        };

        // Update statistics
        stats.total_sent += 1;
        match result.status {
            DeliveryStatus::Delivered => stats.successful_deliveries += 1,
            DeliveryStatus::Failed => stats.failed_deliveries += 1,
            _ => {}
        }

        if let Ok(elapsed) = start_time.elapsed() {
            stats.avg_delivery_time = (stats.avg_delivery_time * (stats.total_sent - 1) as u32 + elapsed) / stats.total_sent as u32;
        }

        stats.last_activity = Some(SystemTime::now());

        Ok(result)
    }

    fn health_check(&self) -> SklResult<ChannelHealth> {
        let stats = self.stats.read().unwrap();

        let score = if stats.total_sent == 0 {
            1.0
        } else {
            stats.successful_deliveries as f64 / stats.total_sent as f64
        };

        let status = if score >= 0.95 {
            ChannelStatus::Healthy
        } else if score >= 0.8 {
            ChannelStatus::Degraded
        } else {
            ChannelStatus::Unavailable
        };

        Ok(ChannelHealth {
            status,
            score,
            last_success: stats.last_activity,
            recent_failures: stats.failed_deliveries as usize,
            avg_delivery_time: stats.avg_delivery_time,
        })
    }

    fn name(&self) -> &str {
        "slack"
    }

    fn supported_severities(&self) -> Vec<SeverityLevel> {
        vec![SeverityLevel::Low, SeverityLevel::Medium, SeverityLevel::High, SeverityLevel::Critical]
    }

    fn supports_batching(&self) -> bool {
        false
    }
}

/// Alert suppression manager
#[derive(Debug)]
pub struct AlertSuppressionManager {
    /// Suppression rules
    rules: Vec<SuppressionRule>,

    /// Active suppressions
    active_suppressions: Arc<RwLock<HashMap<String, ActiveSuppression>>>,

    /// Configuration
    config: AlertSuppressionConfig,
}

/// Active suppression
#[derive(Debug, Clone)]
pub struct ActiveSuppression {
    /// Suppression ID
    pub suppression_id: String,

    /// Rule that created this suppression
    pub rule_name: String,

    /// Start time
    pub started_at: SystemTime,

    /// End time
    pub ends_at: SystemTime,

    /// Suppressed alerts count
    pub suppressed_count: usize,

    /// Suppression reason
    pub reason: String,
}

impl AlertSuppressionManager {
    /// Create new suppression manager
    pub fn new(config: AlertSuppressionConfig) -> Self {
        Self {
            rules: config.rules.clone(),
            active_suppressions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check if alert should be suppressed
    pub fn should_suppress_alert(&self, alert: &ActiveAlert) -> bool {
        if !self.config.enabled {
            return false;
        }

        let active_suppressions = self.active_suppressions.read().unwrap();

        for suppression in active_suppressions.values() {
            if SystemTime::now() <= suppression.ends_at {
                // Check if this suppression applies to the alert
                if self.suppression_applies(suppression, alert) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if suppression applies to alert
    fn suppression_applies(&self, suppression: &ActiveSuppression, alert: &ActiveAlert) -> bool {
        // Find the rule that created this suppression
        for rule in &self.rules {
            if rule.name == suppression.rule_name {
                return self.condition_matches(&rule.condition, alert);
            }
        }
        false
    }

    /// Check if suppression condition matches alert
    fn condition_matches(&self, condition: &SuppressionCondition, alert: &ActiveAlert) -> bool {
        match condition {
            SuppressionCondition::AlertType(alert_type) => {
                alert.rule_name.contains(alert_type)
            }
            SuppressionCondition::Severity(severity) => {
                alert.severity == *severity
            }
            SuppressionCondition::Source(source) => {
                alert.context.session_id.contains(source)
            }
            SuppressionCondition::TimeRange(_) => {
                // Simplified implementation
                true
            }
            SuppressionCondition::Custom(_) => {
                // Simplified implementation
                false
            }
        }
    }

    /// Add temporary suppression
    pub fn add_suppression(&mut self, rule_name: String, duration: Duration, reason: String) -> SklResult<String> {
        let suppression_id = uuid::Uuid::new_v4().to_string();
        let now = SystemTime::now();

        let suppression = ActiveSuppression {
            suppression_id: suppression_id.clone(),
            rule_name,
            started_at: now,
            ends_at: now + duration,
            suppressed_count: 0,
            reason,
        };

        self.active_suppressions.write().unwrap().insert(suppression_id.clone(), suppression);

        Ok(suppression_id)
    }

    /// Remove suppression
    pub fn remove_suppression(&mut self, suppression_id: &str) -> SklResult<bool> {
        Ok(self.active_suppressions.write().unwrap().remove(suppression_id).is_some())
    }

    /// Get active suppressions
    pub fn get_active_suppressions(&self) -> Vec<ActiveSuppression> {
        let now = SystemTime::now();
        self.active_suppressions.read().unwrap()
            .values()
            .filter(|s| s.ends_at > now)
            .cloned()
            .collect()
    }

    /// Cleanup expired suppressions
    pub fn cleanup_expired(&mut self) -> SklResult<usize> {
        let now = SystemTime::now();
        let mut active_suppressions = self.active_suppressions.write().unwrap();
        let initial_count = active_suppressions.len();

        active_suppressions.retain(|_, suppression| suppression.ends_at > now);

        Ok(initial_count - active_suppressions.len())
    }
}

/// Alert escalation manager
#[derive(Debug)]
pub struct AlertEscalationManager {
    /// Escalation policies
    policies: Vec<EscalationPolicy>,

    /// Active escalations
    active_escalations: Arc<RwLock<HashMap<String, ActiveEscalation>>>,

    /// Configuration
    config: EscalationConfig,
}

/// Escalation policy
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// Policy name
    pub name: String,

    /// Escalation levels
    pub levels: Vec<EscalationLevel>,

    /// Policy conditions
    pub conditions: Vec<EscalationCondition>,

    /// Policy enabled
    pub enabled: bool,
}

/// Escalation level
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Level number
    pub level: usize,

    /// Escalation delay
    pub delay: Duration,

    /// Target channels
    pub channels: Vec<String>,

    /// Required acknowledgment
    pub requires_ack: bool,

    /// Repeat interval
    pub repeat_interval: Option<Duration>,
}

/// Escalation condition
#[derive(Debug, Clone)]
pub enum EscalationCondition {
    Severity(SeverityLevel),
    Duration(Duration),
    UnacknowledgedFor(Duration),
    Custom(String),
}

/// Active escalation
#[derive(Debug, Clone)]
pub struct ActiveEscalation {
    /// Escalation ID
    pub escalation_id: String,

    /// Alert ID
    pub alert_id: String,

    /// Policy name
    pub policy_name: String,

    /// Current level
    pub current_level: usize,

    /// Started at
    pub started_at: SystemTime,

    /// Next escalation time
    pub next_escalation: SystemTime,

    /// Escalation count
    pub escalation_count: usize,
}

/// Escalation configuration
#[derive(Debug, Clone)]
pub struct EscalationConfig {
    /// Enable escalation
    pub enabled: bool,

    /// Default policy
    pub default_policy: String,

    /// Maximum escalation level
    pub max_level: usize,

    /// Escalation timeout
    pub timeout: Duration,
}

impl AlertEscalationManager {
    /// Create new escalation manager
    pub fn new(policies: Vec<EscalationPolicy>, config: EscalationConfig) -> Self {
        Self {
            policies,
            active_escalations: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Start escalation for alert
    pub fn start_escalation(&mut self, alert: &ActiveAlert) -> SklResult<Option<String>> {
        if !self.config.enabled {
            return Ok(None);
        }

        // Find applicable escalation policy
        let policy = self.find_applicable_policy(alert)?;
        if let Some(policy) = policy {
            let escalation_id = uuid::Uuid::new_v4().to_string();
            let now = SystemTime::now();

            let escalation = ActiveEscalation {
                escalation_id: escalation_id.clone(),
                alert_id: alert.alert_id.clone(),
                policy_name: policy.name.clone(),
                current_level: 0,
                started_at: now,
                next_escalation: now + policy.levels[0].delay,
                escalation_count: 0,
            };

            self.active_escalations.write().unwrap().insert(escalation_id.clone(), escalation);

            Ok(Some(escalation_id))
        } else {
            Ok(None)
        }
    }

    /// Find applicable escalation policy
    fn find_applicable_policy(&self, alert: &ActiveAlert) -> SklResult<Option<&EscalationPolicy>> {
        for policy in &self.policies {
            if !policy.enabled {
                continue;
            }

            let mut matches = true;
            for condition in &policy.conditions {
                if !self.condition_matches(condition, alert) {
                    matches = false;
                    break;
                }
            }

            if matches {
                return Ok(Some(policy));
            }
        }

        Ok(None)
    }

    /// Check if escalation condition matches
    fn condition_matches(&self, condition: &EscalationCondition, alert: &ActiveAlert) -> bool {
        match condition {
            EscalationCondition::Severity(severity) => alert.severity == *severity,
            EscalationCondition::Duration(duration) => {
                SystemTime::now().duration_since(alert.started_at).unwrap_or_default() >= *duration
            }
            EscalationCondition::UnacknowledgedFor(duration) => {
                !alert.acknowledged && SystemTime::now().duration_since(alert.started_at).unwrap_or_default() >= *duration
            }
            EscalationCondition::Custom(_) => false, // Simplified
        }
    }

    /// Process escalations
    pub fn process_escalations(&mut self) -> SklResult<Vec<EscalationAction>> {
        let now = SystemTime::now();
        let mut actions = Vec::new();
        let mut escalations = self.active_escalations.write().unwrap();

        for escalation in escalations.values_mut() {
            if now >= escalation.next_escalation {
                if let Some(policy) = self.policies.iter().find(|p| p.name == escalation.policy_name) {
                    if escalation.current_level < policy.levels.len() {
                        let level = &policy.levels[escalation.current_level];

                        actions.push(EscalationAction {
                            escalation_id: escalation.escalation_id.clone(),
                            alert_id: escalation.alert_id.clone(),
                            level: escalation.current_level,
                            channels: level.channels.clone(),
                            requires_ack: level.requires_ack,
                        });

                        escalation.escalation_count += 1;
                        escalation.current_level += 1;

                        // Set next escalation time
                        if escalation.current_level < policy.levels.len() {
                            escalation.next_escalation = now + policy.levels[escalation.current_level].delay;
                        } else if let Some(repeat_interval) = level.repeat_interval {
                            escalation.next_escalation = now + repeat_interval;
                        }
                    }
                }
            }
        }

        Ok(actions)
    }

    /// Stop escalation
    pub fn stop_escalation(&mut self, alert_id: &str) -> SklResult<bool> {
        let mut escalations = self.active_escalations.write().unwrap();
        let keys_to_remove: Vec<String> = escalations
            .iter()
            .filter(|(_, escalation)| escalation.alert_id == alert_id)
            .map(|(key, _)| key.clone())
            .collect();

        for key in &keys_to_remove {
            escalations.remove(key);
        }

        Ok(!keys_to_remove.is_empty())
    }
}

/// Escalation action to be performed
#[derive(Debug, Clone)]
pub struct EscalationAction {
    /// Escalation ID
    pub escalation_id: String,

    /// Alert ID
    pub alert_id: String,

    /// Escalation level
    pub level: usize,

    /// Target channels
    pub channels: Vec<String>,

    /// Requires acknowledgment
    pub requires_ack: bool,
}

/// Alert manager statistics
#[derive(Debug, Clone)]
pub struct AlertManagerStats {
    /// Total alerts processed
    pub total_alerts: u64,

    /// Active alerts count
    pub active_alerts: usize,

    /// Resolved alerts count
    pub resolved_alerts: u64,

    /// Suppressed alerts count
    pub suppressed_alerts: u64,

    /// Escalated alerts count
    pub escalated_alerts: u64,

    /// Notifications sent
    pub notifications_sent: u64,

    /// Failed notifications
    pub failed_notifications: u64,

    /// Average resolution time
    pub avg_resolution_time: Duration,

    /// Channel health scores
    pub channel_health: HashMap<String, f64>,
}

impl AlertManager {
    /// Create new alert manager
    pub fn new(config: AlertConfig) -> Self {
        let suppression_manager = AlertSuppressionManager::new(config.suppression.clone());
        let escalation_manager = AlertEscalationManager::new(Vec::new(), EscalationConfig {
            enabled: false,
            default_policy: "default".to_string(),
            max_level: 3,
            timeout: Duration::from_secs(3600),
        });

        Self {
            rules: config.rules.clone(),
            channels: HashMap::new(),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            suppression_manager,
            escalation_manager,
            config,
            stats: Arc::new(RwLock::new(AlertManagerStats::default())),
        }
    }

    /// Add notification channel
    pub fn add_channel(&mut self, name: String, channel: Box<dyn NotificationChannel>) {
        self.channels.insert(name, channel);
    }

    /// Evaluate alert rules against metrics
    pub fn evaluate_rules(&mut self, metrics: &[PerformanceMetric], events: &[TaskExecutionEvent]) -> SklResult<Vec<AlertTriggered>> {
        let mut triggered_alerts = Vec::new();

        for rule in &mut self.rules {
            if !rule.config.enabled {
                continue;
            }

            // Check if rule should be evaluated
            if let Some(last_eval) = rule.last_evaluation {
                if SystemTime::now().duration_since(last_eval).unwrap_or_default() < rule.config.evaluation_interval {
                    continue;
                }
            }

            // Evaluate rule condition
            let evaluation_result = self.evaluate_condition(&rule.condition, metrics, events)?;
            rule.last_evaluation = Some(SystemTime::now());
            rule.evaluation_count += 1;

            // Update rule state and trigger alerts
            let state_change = self.update_rule_state(rule, evaluation_result);
            if let Some(alert) = state_change {
                triggered_alerts.push(alert);
            }
        }

        Ok(triggered_alerts)
    }

    /// Evaluate a single alert condition
    fn evaluate_condition(
        &self,
        condition: &AlertCondition,
        metrics: &[PerformanceMetric],
        events: &[TaskExecutionEvent],
    ) -> SklResult<bool> {
        match condition {
            AlertCondition::Threshold { metric_name, operator, threshold, window, aggregation } => {
                let relevant_metrics: Vec<&PerformanceMetric> = metrics
                    .iter()
                    .filter(|m| m.name == *metric_name)
                    .filter(|m| SystemTime::now().duration_since(m.timestamp).unwrap_or_default() <= *window)
                    .collect();

                if relevant_metrics.is_empty() {
                    return Ok(false);
                }

                let aggregated_value = self.aggregate_metric_values(&relevant_metrics, aggregation)?;
                Ok(self.compare_value(aggregated_value, operator, *threshold))
            }

            AlertCondition::Rate { metric_name, rate_threshold, window, direction } => {
                let relevant_metrics: Vec<&PerformanceMetric> = metrics
                    .iter()
                    .filter(|m| m.name == *metric_name)
                    .filter(|m| SystemTime::now().duration_since(m.timestamp).unwrap_or_default() <= *window)
                    .collect();

                if relevant_metrics.len() < 2 {
                    return Ok(false);
                }

                let rate = self.calculate_rate(&relevant_metrics, *window)?;
                match direction {
                    RateDirection::Increasing => Ok(rate > *rate_threshold),
                    RateDirection::Decreasing => Ok(rate < -*rate_threshold),
                    RateDirection::Both => Ok(rate.abs() > *rate_threshold),
                }
            }

            AlertCondition::Event { event_type, severity, count_threshold, window } => {
                let relevant_events = events
                    .iter()
                    .filter(|e| e.event_type == *event_type && e.severity == *severity)
                    .filter(|e| SystemTime::now().duration_since(e.timestamp).unwrap_or_default() <= *window)
                    .count();

                Ok(relevant_events >= *count_threshold)
            }

            AlertCondition::Composite { operator, conditions } => {
                match operator {
                    LogicalOperator::And => {
                        for condition in conditions {
                            if !self.evaluate_condition(condition, metrics, events)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    LogicalOperator::Or => {
                        for condition in conditions {
                            if self.evaluate_condition(condition, metrics, events)? {
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    }
                    LogicalOperator::Not => {
                        if conditions.len() == 1 {
                            Ok(!self.evaluate_condition(&conditions[0], metrics, events)?)
                        } else {
                            Err(SklearsError::InvalidInput("NOT operator requires exactly one condition".to_string()))
                        }
                    }
                }
            }

            _ => {
                // For other condition types, return false (not implemented)
                Ok(false)
            }
        }
    }

    /// Aggregate metric values according to function
    fn aggregate_metric_values(&self, metrics: &[&PerformanceMetric], function: &AggregationFunction) -> SklResult<f64> {
        if metrics.is_empty() {
            return Ok(0.0);
        }

        let values: Vec<f64> = metrics.iter().map(|m| m.value).collect();

        match function {
            AggregationFunction::Mean => Ok(values.iter().sum::<f64>() / values.len() as f64),
            AggregationFunction::Sum => Ok(values.iter().sum()),
            AggregationFunction::Min => Ok(values.iter().cloned().fold(f64::INFINITY, f64::min)),
            AggregationFunction::Max => Ok(values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
            AggregationFunction::Count => Ok(values.len() as f64),
            _ => Ok(0.0), // Simplified for other functions
        }
    }

    /// Compare value using operator
    fn compare_value(&self, value: f64, operator: &ComparisonOperator, threshold: f64) -> bool {
        match operator {
            ComparisonOperator::Greater => value > threshold,
            ComparisonOperator::GreaterEqual => value >= threshold,
            ComparisonOperator::Less => value < threshold,
            ComparisonOperator::LessEqual => value <= threshold,
            ComparisonOperator::Equal => (value - threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (value - threshold).abs() >= f64::EPSILON,
            ComparisonOperator::Between { min, max } => value >= *min && value <= *max,
            ComparisonOperator::NotBetween { min, max } => value < *min || value > *max,
        }
    }

    /// Calculate rate of change
    fn calculate_rate(&self, metrics: &[&PerformanceMetric], _window: Duration) -> SklResult<f64> {
        if metrics.len() < 2 {
            return Ok(0.0);
        }

        // Simplified rate calculation - would need proper time-based rate calculation
        let first_value = metrics.first().unwrap().value;
        let last_value = metrics.last().unwrap().value;

        Ok(last_value - first_value)
    }

    /// Update rule state based on evaluation result
    fn update_rule_state(&mut self, rule: &mut AlertRule, triggered: bool) -> Option<AlertTriggered> {
        match &rule.state {
            AlertRuleState::Normal => {
                if triggered {
                    if rule.config.consecutive_triggers <= 1 {
                        rule.state = AlertRuleState::Triggered;
                        Some(self.create_alert_triggered(rule))
                    } else {
                        rule.state = AlertRuleState::Pending { trigger_count: 1 };
                        None
                    }
                } else {
                    None
                }
            }
            AlertRuleState::Pending { trigger_count } => {
                if triggered {
                    let new_count = trigger_count + 1;
                    if new_count >= rule.config.consecutive_triggers {
                        rule.state = AlertRuleState::Triggered;
                        Some(self.create_alert_triggered(rule))
                    } else {
                        rule.state = AlertRuleState::Pending { trigger_count: new_count };
                        None
                    }
                } else {
                    rule.state = AlertRuleState::Normal;
                    None
                }
            }
            AlertRuleState::Triggered => {
                if !triggered && rule.config.recovery_evaluation {
                    rule.state = AlertRuleState::Recovery;
                }
                None
            }
            AlertRuleState::Recovery => {
                if triggered {
                    rule.state = AlertRuleState::Triggered;
                    None
                } else {
                    rule.state = AlertRuleState::Normal;
                    // Would create alert resolved event here
                    None
                }
            }
            AlertRuleState::Disabled => None,
        }
    }

    /// Create alert triggered event
    fn create_alert_triggered(&self, rule: &AlertRule) -> AlertTriggered {
        AlertTriggered {
            rule_id: rule.id.clone(),
            rule_name: rule.name.clone(),
            severity: rule.severity.clone(),
            message: format!("Alert triggered: {}", rule.description),
            triggered_at: SystemTime::now(),
            context: AlertContext {
                session_id: "unknown".to_string(),
                metric_values: HashMap::new(),
                related_events: Vec::new(),
                context_data: HashMap::new(),
            },
        }
    }

    /// Process triggered alert
    pub fn process_alert(&mut self, triggered: AlertTriggered) -> SklResult<()> {
        // Check if alert should be suppressed
        let alert = ActiveAlert {
            alert_id: uuid::Uuid::new_v4().to_string(),
            rule_id: triggered.rule_id.clone(),
            rule_name: triggered.rule_name,
            message: triggered.message,
            severity: triggered.severity,
            started_at: triggered.triggered_at,
            updated_at: triggered.triggered_at,
            state: AlertState::Firing,
            trigger_value: None,
            context: triggered.context,
            notifications: Vec::new(),
            escalation_level: 0,
            acknowledged: false,
            acknowledged_by: None,
            tags: HashMap::new(),
        };

        if self.suppression_manager.should_suppress_alert(&alert) {
            self.stats.write().unwrap().suppressed_alerts += 1;
            return Ok(());
        }

        // Send notifications
        let rule = self.rules.iter().find(|r| r.id == triggered.rule_id);
        if let Some(rule) = rule {
            for channel_name in &rule.channels {
                if let Some(channel) = self.channels.get_mut(channel_name) {
                    match channel.send_notification(&alert) {
                        Ok(result) => {
                            self.stats.write().unwrap().notifications_sent += 1;
                            // Would record notification status here
                        }
                        Err(_) => {
                            self.stats.write().unwrap().failed_notifications += 1;
                        }
                    }
                }
            }
        }

        // Store active alert
        self.active_alerts.write().unwrap().insert(alert.alert_id.clone(), alert);

        // Start escalation if configured
        // self.escalation_manager.start_escalation(&alert)?;

        self.stats.write().unwrap().total_alerts += 1;

        Ok(())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self, session_id: &str) -> SklResult<Vec<ActiveAlert>> {
        let active_alerts = self.active_alerts.read().unwrap();
        let alerts: Vec<ActiveAlert> = active_alerts
            .values()
            .filter(|alert| alert.context.session_id == session_id)
            .cloned()
            .collect();
        Ok(alerts)
    }

    /// Get manager statistics
    pub fn get_stats(&self) -> AlertManagerStats {
        let mut stats = self.stats.read().unwrap().clone();
        stats.active_alerts = self.active_alerts.read().unwrap().len();

        // Update channel health scores
        for (name, channel) in &self.channels {
            if let Ok(health) = channel.health_check() {
                stats.channel_health.insert(name.clone(), health.score);
            }
        }

        stats
    }
}

/// Alert triggered event
#[derive(Debug, Clone)]
pub struct AlertTriggered {
    /// Rule ID
    pub rule_id: String,

    /// Rule name
    pub rule_name: String,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Alert message
    pub message: String,

    /// Trigger timestamp
    pub triggered_at: SystemTime,

    /// Alert context
    pub context: AlertContext,
}

/// Alert management system
pub struct AlertManagementSystem {
    /// Alert manager
    manager: AlertManager,
}

impl AlertManagementSystem {
    /// Create new alert management system
    pub fn new(config: AlertConfig) -> Self {
        Self {
            manager: AlertManager::new(config),
        }
    }

    /// Initialize session
    pub fn initialize_session(&mut self, session_id: &str, config: &AlertConfig) -> SklResult<()> {
        // Initialize session-specific alert configuration
        Ok(())
    }

    /// Get active alerts for session
    pub fn get_active_alerts(&self, session_id: &str) -> SklResult<Vec<ActiveAlert>> {
        self.manager.get_active_alerts(session_id)
    }

    /// Finalize session
    pub fn finalize_session(&mut self, session_id: &str) -> SklResult<Vec<AlertRecord>> {
        // Convert active alerts to alert records for final report
        let active_alerts = self.manager.get_active_alerts(session_id)?;
        let records: Vec<AlertRecord> = active_alerts
            .into_iter()
            .map(|alert| AlertRecord {
                alert_id: alert.alert_id,
                rule_name: alert.rule_name,
                severity: alert.severity,
                triggered_at: alert.started_at,
                resolved_at: None,
                message: alert.message,
                metric_values: alert.context.metric_values,
            })
            .collect();
        Ok(records)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AlertConfig) -> SklResult<()> {
        // Update alert manager configuration
        Ok(())
    }

    /// Get health status
    pub fn get_health_status(&self) -> SklResult<ComponentHealth> {
        let stats = self.manager.get_stats();

        let mut health_score = 1.0;
        let mut issues = Vec::new();

        // Check for failed notifications
        if stats.failed_notifications > 0 {
            let failure_rate = stats.failed_notifications as f64 / stats.notifications_sent as f64;
            if failure_rate > 0.1 {
                health_score -= 0.3;
                issues.push("High notification failure rate".to_string());
            }
        }

        // Check channel health
        for (channel, score) in &stats.channel_health {
            if *score < 0.8 {
                health_score -= 0.2;
                issues.push(format!("Channel {} degraded", channel));
            }
        }

        let status = if health_score >= 0.8 {
            HealthStatus::Healthy
        } else if health_score >= 0.5 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };

        Ok(ComponentHealth {
            component: "alert_management".to_string(),
            status,
            score: health_score,
            last_check: SystemTime::now(),
            issues,
        })
    }
}

impl Default for ChannelStats {
    fn default() -> Self {
        Self {
            total_sent: 0,
            successful_deliveries: 0,
            failed_deliveries: 0,
            avg_delivery_time: Duration::ZERO,
            last_activity: None,
        }
    }
}

impl Default for AlertManagerStats {
    fn default() -> Self {
        Self {
            total_alerts: 0,
            active_alerts: 0,
            resolved_alerts: 0,
            suppressed_alerts: 0,
            escalated_alerts: 0,
            notifications_sent: 0,
            failed_notifications: 0,
            avg_resolution_time: Duration::ZERO,
            channel_health: HashMap::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_rule_evaluation() {
        let rule = AlertRule {
            id: "test_rule".to_string(),
            name: "High CPU Usage".to_string(),
            description: "CPU usage above 80%".to_string(),
            condition: AlertCondition::Threshold {
                metric_name: "cpu_usage".to_string(),
                operator: ComparisonOperator::Greater,
                threshold: 80.0,
                window: Duration::from_secs(60),
                aggregation: AggregationFunction::Mean,
            },
            severity: SeverityLevel::Critical,
            channels: vec!["email".to_string()],
            config: AlertRuleConfig {
                enabled: true,
                evaluation_interval: Duration::from_secs(30),
                evaluation_timeout: Duration::from_secs(10),
                consecutive_triggers: 1,
                recovery_evaluation: true,
                auto_resolve_after: None,
                tags: HashMap::new(),
            },
            state: AlertRuleState::Normal,
            last_evaluation: None,
            evaluation_count: 0,
        };

        assert!(matches!(rule.state, AlertRuleState::Normal));
        assert_eq!(rule.severity, SeverityLevel::Critical);
    }

    #[test]
    fn test_email_channel() {
        let config = EmailChannelConfig {
            smtp_server: "smtp.example.com".to_string(),
            smtp_port: 587,
            username: "test@example.com".to_string(),
            password: "password".to_string(),
            from_email: "alerts@example.com".to_string(),
            default_recipients: vec!["admin@example.com".to_string()],
            use_tls: true,
            timeout: Duration::from_secs(30),
            template: EmailTemplate {
                subject: "Alert: {alert_name}".to_string(),
                body_html: "<h1>{alert_name}</h1><p>{message}</p>".to_string(),
                body_text: "{alert_name}: {message}".to_string(),
                include_details: true,
            },
        };

        let mut channel = EmailChannel::new(config);

        let alert = ActiveAlert {
            alert_id: "test_alert".to_string(),
            rule_id: "test_rule".to_string(),
            rule_name: "Test Alert".to_string(),
            message: "Test message".to_string(),
            severity: SeverityLevel::High,
            started_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            state: AlertState::Firing,
            trigger_value: Some(85.0),
            context: AlertContext {
                session_id: "test_session".to_string(),
                metric_values: HashMap::new(),
                related_events: Vec::new(),
                context_data: HashMap::new(),
            },
            notifications: Vec::new(),
            escalation_level: 0,
            acknowledged: false,
            acknowledged_by: None,
            tags: HashMap::new(),
        };

        let result = channel.send_notification(&alert).unwrap();
        assert!(matches!(result.status, DeliveryStatus::Delivered));
        assert_eq!(channel.name(), "email");
    }

    #[test]
    fn test_alert_suppression() {
        let config = AlertSuppressionConfig {
            enabled: true,
            rules: vec![
                SuppressionRule {
                    name: "Maintenance Window".to_string(),
                    condition: SuppressionCondition::AlertType("cpu".to_string()),
                    duration: Duration::from_secs(3600),
                    priority: 1,
                }
            ],
            default_duration: Duration::from_secs(300),
        };

        let mut suppression_manager = AlertSuppressionManager::new(config);

        let alert = ActiveAlert {
            alert_id: "test_alert".to_string(),
            rule_id: "cpu_rule".to_string(),
            rule_name: "CPU Alert".to_string(),
            message: "High CPU usage".to_string(),
            severity: SeverityLevel::High,
            started_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            state: AlertState::Firing,
            trigger_value: Some(85.0),
            context: AlertContext {
                session_id: "test_session".to_string(),
                metric_values: HashMap::new(),
                related_events: Vec::new(),
                context_data: HashMap::new(),
            },
            notifications: Vec::new(),
            escalation_level: 0,
            acknowledged: false,
            acknowledged_by: None,
            tags: HashMap::new(),
        };

        // Initially should not be suppressed
        assert!(!suppression_manager.should_suppress_alert(&alert));

        // Add suppression
        let suppression_id = suppression_manager.add_suppression(
            "cpu".to_string(),
            Duration::from_secs(3600),
            "Maintenance window".to_string(),
        ).unwrap();

        // Now should be suppressed
        assert!(suppression_manager.should_suppress_alert(&alert));

        // Remove suppression
        assert!(suppression_manager.remove_suppression(&suppression_id).unwrap());

        // Should not be suppressed anymore
        assert!(!suppression_manager.should_suppress_alert(&alert));
    }

    #[test]
    fn test_alert_manager() {
        let config = AlertConfig {
            enabled: true,
            rules: Vec::new(),
            channels: Vec::new(),
            aggregation: AlertAggregationConfig {
                enabled: false,
                window: Duration::from_secs(60),
                strategy: AlertAggregationStrategy::Count,
                max_alerts: 10,
            },
            suppression: AlertSuppressionConfig {
                enabled: false,
                rules: Vec::new(),
                default_duration: Duration::from_secs(300),
            },
        };

        let mut manager = AlertManager::new(config);

        // Add email channel
        let email_config = EmailChannelConfig {
            smtp_server: "smtp.example.com".to_string(),
            smtp_port: 587,
            username: "test@example.com".to_string(),
            password: "password".to_string(),
            from_email: "alerts@example.com".to_string(),
            default_recipients: vec!["admin@example.com".to_string()],
            use_tls: true,
            timeout: Duration::from_secs(30),
            template: EmailTemplate {
                subject: "Alert: {alert_name}".to_string(),
                body_html: "<h1>{alert_name}</h1><p>{message}</p>".to_string(),
                body_text: "{alert_name}: {message}".to_string(),
                include_details: true,
            },
        };

        let email_channel: Box<dyn NotificationChannel> = Box::new(EmailChannel::new(email_config));
        manager.add_channel("email".to_string(), email_channel);

        let stats = manager.get_stats();
        assert_eq!(stats.total_alerts, 0);
        assert_eq!(stats.active_alerts, 0);
    }
}