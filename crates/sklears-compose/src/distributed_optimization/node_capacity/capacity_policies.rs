use crate::distributed_optimization::core_types::*;
use super::real_time_monitoring::{AlertCondition, AlertSeverity, ComparisonOperator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Capacity policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityPolicies {
    pub scaling_policies: Vec<ScalingPolicy>,
    pub resource_limits: HashMap<String, ResourceLimit>,
    pub capacity_thresholds: CapacityThresholds,
    pub emergency_procedures: EmergencyProcedures,
}

/// Scaling policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    pub policy_id: String,
    pub scaling_trigger: ScalingTrigger,
    pub scaling_action: ScalingAction,
    pub cooldown_period: Duration,
    pub min_capacity: f64,
    pub max_capacity: f64,
}

/// Scaling trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingTrigger {
    pub metric_name: String,
    pub threshold: f64,
    pub comparison: ComparisonOperator,
    pub evaluation_period: Duration,
    pub consecutive_periods: u32,
}

/// Scaling action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    ScaleUp(f64),
    ScaleDown(f64),
    AddNodes(u32),
    RemoveNodes(u32),
    Custom(String),
}

/// Resource limit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimit {
    pub resource_type: String,
    pub soft_limit: f64,
    pub hard_limit: f64,
    pub enforcement_action: EnforcementAction,
}

/// Enforcement action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementAction {
    Warn,
    Throttle,
    Reject,
    Queue,
    Preempt,
    Custom(String),
}

/// Capacity thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityThresholds {
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub emergency_threshold: f64,
    pub threshold_metrics: Vec<String>,
}

/// Emergency procedures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyProcedures {
    pub enabled: bool,
    pub emergency_contacts: Vec<String>,
    pub escalation_procedures: Vec<EscalationStep>,
    pub automatic_actions: Vec<EmergencyAction>,
}

/// Escalation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationStep {
    pub step_id: String,
    pub escalation_delay: Duration,
    pub notification_method: String,
    pub escalation_criteria: String,
}

/// Emergency action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyAction {
    ScaleOut,
    LoadShedding,
    GracefulDegradation,
    FailoverActivation,
    ResourceReallocation,
    Custom(String),
}

/// Capacity alert manager
pub struct CapacityAlertManager {
    pub alert_rules: Vec<AlertRule>,
    pub active_alerts: Vec<ActiveAlert>,
    pub alert_history: Vec<AlertEvent>,
    pub notification_channels: HashMap<String, NotificationChannel>,
    pub alert_correlations: Vec<AlertCorrelation>,
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub rule_id: String,
    pub rule_name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub notification_channels: Vec<String>,
    pub suppress_duration: Duration,
}

/// Active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub rule_id: String,
    pub node_id: NodeId,
    pub alert_time: SystemTime,
    pub current_value: f64,
    pub threshold_value: f64,
    pub alert_message: String,
}

/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub event_id: String,
    pub alert_id: String,
    pub event_type: AlertEventType,
    pub event_time: SystemTime,
    pub event_details: HashMap<String, String>,
}

/// Alert event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertEventType {
    Triggered,
    Resolved,
    Acknowledged,
    Escalated,
    Suppressed,
}

/// Notification channel
pub struct NotificationChannel {
    pub channel_id: String,
    pub channel_type: NotificationChannelType,
    pub configuration: HashMap<String, String>,
    pub enabled: bool,
    pub rate_limit: Option<RateLimit>,
}

/// Notification channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Email,
    SMS,
    Slack,
    PagerDuty,
    Webhook,
    SNMP,
    Custom(String),
}

/// Rate limit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub max_notifications: u32,
    pub time_window: Duration,
    pub burst_limit: u32,
}

/// Alert correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCorrelation {
    pub correlation_id: String,
    pub primary_alert: String,
    pub related_alerts: Vec<String>,
    pub correlation_strength: f64,
    pub correlation_type: CorrelationType,
}

/// Correlation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    Causal,
    Temporal,
    Spatial,
    Thematic,
    Statistical,
    Custom(String),
}

impl CapacityAlertManager {
    pub fn new() -> Self {
        Self {
            alert_rules: Vec::new(),
            active_alerts: Vec::new(),
            alert_history: Vec::new(),
            notification_channels: HashMap::new(),
            alert_correlations: Vec::new(),
        }
    }
}

impl Default for CapacityPolicies {
    fn default() -> Self {
        Self {
            scaling_policies: Vec::new(),
            resource_limits: HashMap::new(),
            capacity_thresholds: CapacityThresholds {
                warning_threshold: 70.0,
                critical_threshold: 85.0,
                emergency_threshold: 95.0,
                threshold_metrics: vec!["cpu".to_string(), "memory".to_string(), "storage".to_string()],
            },
            emergency_procedures: EmergencyProcedures {
                enabled: true,
                emergency_contacts: Vec::new(),
                escalation_procedures: Vec::new(),
                automatic_actions: Vec::new(),
            },
        }
    }
}