//! Alert Configuration
//!
//! This module contains all configuration structures related to alerting, notifications,
//! and escalation policies. It provides comprehensive control over how alerts are
//! generated, routed, and managed in the monitoring system.

use std::collections::HashMap;
use std::time::Duration;
use super::config_core::{SeverityLevel, ComparisonOperator};

/// Alert configuration
///
/// Controls all aspects of alerting including alert rules, notification channels,
/// aggregation policies, and suppression mechanisms. Provides comprehensive
/// alerting capabilities for proactive monitoring and incident response.
///
/// # Architecture
///
/// The alerting system follows a flexible, rule-based architecture:
///
/// ```text
/// Alert System
/// ├── Alert Rules (conditions, thresholds, evaluations)
/// ├── Alert Channels (notifications, routing, delivery)
/// ├── Alert Aggregation (grouping, deduplication)
/// ├── Alert Suppression (maintenance, noise reduction)
/// ├── Escalation Policies (severity-based routing)
/// └── Alert Management (acknowledgment, resolution)
/// ```
///
/// # Alert Flow
///
/// Alerts follow a structured flow from detection to resolution:
/// 1. **Detection**: Alert rules evaluate conditions against metrics/events
/// 2. **Aggregation**: Related alerts are grouped and deduplicated
/// 3. **Routing**: Alerts are routed to appropriate channels based on rules
/// 4. **Notification**: Alerts are sent through configured channels
/// 5. **Escalation**: Unacknowledged alerts escalate according to policies
/// 6. **Management**: Alerts can be acknowledged, annotated, and resolved
///
/// # Usage Examples
///
/// ## Production Alerting Configuration
/// ```rust
/// use sklears_compose::monitoring_config::AlertConfig;
///
/// let config = AlertConfig::production();
/// ```
///
/// ## Development Alerting Configuration
/// ```rust
/// let config = AlertConfig::development();
/// ```
///
/// ## Custom Alert Rules
/// ```rust
/// let config = AlertConfig::custom_rules(vec![
///     AlertRule::threshold("cpu_high", "cpu_usage", 80.0),
///     AlertRule::rate("error_spike", "error_rate", 0.05),
/// ]);
/// ```
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts
    ///
    /// Global switch to enable or disable all alerting functionality.
    /// When disabled, no alerts will be generated or sent.
    pub enabled: bool,

    /// Alert rules
    ///
    /// List of rules that define when and how alerts should be generated.
    /// Rules are evaluated continuously against incoming metrics and events.
    pub rules: Vec<AlertRule>,

    /// Alert channels
    ///
    /// List of notification channels for delivering alerts.
    /// Multiple channels can be configured for redundancy and different use cases.
    pub channels: Vec<AlertChannel>,

    /// Alert aggregation configuration
    ///
    /// Controls how alerts are grouped, deduplicated, and batch-processed
    /// to reduce noise and improve notification efficiency.
    pub aggregation: AlertAggregationConfig,

    /// Alert suppression configuration
    ///
    /// Controls when alerts should be suppressed to reduce noise during
    /// maintenance windows or known issues.
    pub suppression: AlertSuppressionConfig,

    /// Escalation policies
    ///
    /// Defines how alerts should escalate based on severity, time, and acknowledgment status.
    pub escalation: EscalationConfig,

    /// Alert management settings
    ///
    /// Controls alert lifecycle management including acknowledgment, annotation, and resolution.
    pub management: AlertManagementConfig,
}

/// Alert rules
///
/// Defines conditions that trigger alerts when met. Rules can be based on
/// thresholds, rates of change, anomalies, or complex expressions.
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,

    pub description: String,

    pub condition: AlertCondition,

    pub severity: SeverityLevel,

    pub channels: Vec<String>,

    pub enabled: bool,

    pub evaluation_interval: Duration,

    pub metadata: HashMap<String, String>,

    pub dependencies: Vec<String>,

    pub cooldown: Option<Duration>,
}

/// Alert conditions
///
/// Defines the various types of conditions that can trigger alerts.
/// Each condition type has specific parameters and evaluation logic.
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold-based condition
    ///
    /// Triggers when a metric crosses a specified threshold for a given duration.
    Threshold {
        /// Metric name to evaluate
        metric: String,
        /// Comparison operator (>, >=, <, <=, ==, !=)
        operator: ComparisonOperator,
        /// Threshold value
        value: f64,
        /// Duration threshold must be exceeded
        duration: Duration,
        /// Optional metric labels for filtering
        labels: HashMap<String, String>,
    },

    /// Rate-based condition
    ///
    /// Triggers when the rate of change of a metric exceeds a threshold.
    Rate {
        /// Metric name to evaluate
        metric: String,
        /// Rate threshold (change per unit time)
        rate_threshold: f64,
        /// Time window for rate calculation
        window: Duration,
        /// Direction of rate change (increase/decrease/both)
        direction: RateDirection,
    },

    /// Anomaly-based condition
    ///
    /// Triggers when a metric value is detected as anomalous using statistical methods.
    Anomaly {
        /// Metric name to evaluate
        metric: String,
        /// Anomaly detection algorithm
        algorithm: AnomalyDetectionAlgorithm,
        /// Sensitivity level (0.0 to 1.0)
        sensitivity: f64,
        /// Training window for baseline establishment
        training_window: Duration,
    },

    /// Complex condition using expressions
    ///
    /// Triggers based on complex expressions involving multiple metrics.
    Complex {
        /// Boolean expression to evaluate
        expression: String,
        /// Variable mappings (variable name -> metric name)
        variables: HashMap<String, String>,
        /// Expression evaluation timeout
        timeout: Duration,
    },

    /// Composite condition
    ///
    /// Triggers based on combinations of other conditions.
    Composite {
        /// Logical operator (AND, OR, XOR)
        operator: LogicalOperator,
        /// Sub-conditions to evaluate
        conditions: Vec<AlertCondition>,
        /// Minimum number of conditions that must be true
        min_conditions: Option<usize>,
    },

    /// Time-based condition
    ///
    /// Triggers based on time patterns or schedules.
    Time {
        /// Time pattern (cron-like expression)
        pattern: String,
        /// Time zone for evaluation
        timezone: String,
        /// Additional metric conditions
        metric_conditions: Vec<AlertCondition>,
    },
}

/// Rate change directions for rate-based conditions
#[derive(Debug, Clone)]
pub enum RateDirection {
    /// Rate increase only
    Increase,
    /// Rate decrease only
    Decrease,
    /// Both increase and decrease
    Both,
}

/// Anomaly detection algorithms for alert conditions
#[derive(Debug, Clone)]
pub enum AnomalyDetectionAlgorithm {
    /// Z-score based detection
    ZScore { threshold: f64 },
    /// Isolation Forest algorithm
    IsolationForest,
    /// One-Class SVM
    OneClassSvm,
    /// Moving average with standard deviation
    MovingAverage { window: u32, threshold: f64 },
    /// Custom algorithm
    Custom { name: String, config: HashMap<String, f64> },
}

/// Logical operators for composite conditions
#[derive(Debug, Clone)]
pub enum LogicalOperator {
    /// All conditions must be true
    And,
    /// At least one condition must be true
    Or,
    /// Exactly one condition must be true
    Xor,
    /// None of the conditions must be true
    Not,
}

/// Alert channels
///
/// Defines how and where alerts are delivered. Multiple channel types
/// are supported for different notification requirements.
#[derive(Debug, Clone)]
pub enum AlertChannel {
    /// Email notifications
    Email {
        /// Channel name
        name: String,
        /// SMTP server configuration
        smtp_config: SmtpConfig,
        /// Recipient addresses
        recipients: Vec<String>,
        /// Email template settings
        template: EmailTemplate,
    },

    /// Slack notifications
    Slack {
        /// Channel name
        name: String,
        /// Slack webhook URL
        webhook_url: String,
        /// Target channel or user
        channel: String,
        /// Message formatting options
        formatting: SlackFormatting,
    },

    /// PagerDuty integration
    PagerDuty {
        /// Channel name
        name: String,
        /// PagerDuty integration key
        integration_key: String,
        /// Service details
        service: PagerDutyService,
    },

    /// Microsoft Teams notifications
    Teams {
        /// Channel name
        name: String,
        /// Teams webhook URL
        webhook_url: String,
        /// Message theme and formatting
        theme: TeamsTheme,
    },

    /// Discord notifications
    Discord {
        /// Channel name
        name: String,
        /// Discord webhook URL
        webhook_url: String,
        /// Message formatting
        formatting: DiscordFormatting,
    },

    /// SMS notifications
    Sms {
        /// Channel name
        name: String,
        /// SMS service configuration
        service_config: SmsConfig,
        /// Phone numbers
        phone_numbers: Vec<String>,
    },

    /// Webhook notifications
    Webhook {
        /// Channel name
        name: String,
        /// Target URL
        url: String,
        /// HTTP method
        method: String,
        /// Request headers
        headers: HashMap<String, String>,
        /// Authentication settings
        auth: Option<WebhookAuth>,
    },

    /// Custom notification channel
    Custom {
        /// Channel name
        name: String,
        /// Channel type identifier
        channel_type: String,
        /// Configuration parameters
        config: HashMap<String, String>,
    },
}

/// SMTP configuration for email alerts
#[derive(Debug, Clone)]
pub struct SmtpConfig {
    /// SMTP server hostname
    pub host: String,
    /// SMTP server port
    pub port: u16,
    /// Use TLS encryption
    pub use_tls: bool,
    /// Authentication credentials
    pub auth: Option<SmtpAuth>,
    /// Sender email address
    pub from: String,
}

/// SMTP authentication
#[derive(Debug, Clone)]
pub struct SmtpAuth {
    /// Username
    pub username: String,
    /// Password
    pub password: String,
}

/// Email template configuration
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

/// Slack message formatting
#[derive(Debug, Clone)]
pub struct SlackFormatting {
    /// Use rich formatting (blocks)
    pub rich_formatting: bool,
    /// Include alert context
    pub include_context: bool,
    /// Color scheme for different severities
    pub colors: HashMap<String, String>,
}

/// PagerDuty service configuration
#[derive(Debug, Clone)]
pub struct PagerDutyService {
    /// Service name
    pub name: String,
    /// Service ID
    pub id: String,
    /// Escalation policy
    pub escalation_policy: String,
}

/// Microsoft Teams theme
#[derive(Debug, Clone)]
pub struct TeamsTheme {
    /// Theme color
    pub color: String,
    /// Include summary card
    pub summary_card: bool,
    /// Activity format
    pub activity_format: String,
}

/// Discord message formatting
#[derive(Debug, Clone)]
pub struct DiscordFormatting {
    /// Use embeds
    pub use_embeds: bool,
    /// Include timestamp
    pub include_timestamp: bool,
    /// Color for different severities
    pub colors: HashMap<String, u32>,
}

/// SMS service configuration
#[derive(Debug, Clone)]
pub struct SmsConfig {
    /// SMS service provider
    pub provider: String,
    /// API key or credentials
    pub credentials: String,
    /// Sender ID
    pub sender_id: String,
}

/// Webhook authentication
#[derive(Debug, Clone)]
pub enum WebhookAuth {
    /// Basic authentication
    Basic { username: String, password: String },
    /// Bearer token
    Bearer { token: String },
    /// API key
    ApiKey { key: String, header: String },
    /// Custom authentication
    Custom { auth_type: String, config: HashMap<String, String> },
}

/// Alert aggregation configuration
///
/// Controls how alerts are grouped and processed to reduce noise
/// and improve notification efficiency.
#[derive(Debug, Clone)]
pub struct AlertAggregationConfig {
    /// Enable alert aggregation
    pub enabled: bool,

    /// Aggregation window size
    ///
    /// Time window for grouping related alerts together.
    pub window_size: Duration,

    /// Grouping keys
    ///
    /// Alert attributes used for grouping (e.g., service, host, severity).
    pub grouping_keys: Vec<String>,

    /// Maximum alerts per group
    ///
    /// Limit on the number of alerts in a single aggregated notification.
    pub max_alerts_per_group: usize,

    /// Aggregation strategies
    ///
    /// Different strategies for combining alerts within a group.
    pub strategies: Vec<AggregationStrategy>,

    /// Deduplication settings
    ///
    /// Controls how duplicate alerts are handled.
    pub deduplication: DeduplicationConfig,
}

/// Alert aggregation strategies
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// Count-based aggregation (send summary with count)
    Count,
    /// Time-based aggregation (wait for window to close)
    TimeBased,
    /// Severity-based aggregation (group by severity)
    SeverityBased,
    /// Service-based aggregation (group by service/component)
    ServiceBased,
    /// Custom aggregation strategy
    Custom { strategy_name: String, config: HashMap<String, String> },
}

/// Alert deduplication configuration
#[derive(Debug, Clone)]
pub struct DeduplicationConfig {
    /// Enable deduplication
    pub enabled: bool,

    /// Deduplication keys
    ///
    /// Alert attributes used for identifying duplicates.
    pub dedup_keys: Vec<String>,

    /// Deduplication window
    ///
    /// Time window for considering alerts as duplicates.
    pub window: Duration,

    /// Action for duplicate alerts
    pub duplicate_action: DuplicateAction,
}

/// Actions for handling duplicate alerts
#[derive(Debug, Clone)]
pub enum DuplicateAction {
    /// Discard duplicate alerts
    Discard,
    /// Update existing alert with new information
    Update,
    /// Increment counter on existing alert
    Count,
    /// Extend the duration of existing alert
    Extend,
}

/// Alert suppression configuration
///
/// Controls when alerts should be suppressed to reduce noise during
/// maintenance windows or known issues.
#[derive(Debug, Clone)]
pub struct AlertSuppressionConfig {
    /// Enable alert suppression
    pub enabled: bool,

    /// Suppression rules
    ///
    /// Rules that define when alerts should be suppressed.
    pub rules: Vec<SuppressionRule>,

    /// Global suppression settings
    pub global: GlobalSuppressionConfig,

    /// Maintenance window configuration
    pub maintenance_windows: Vec<MaintenanceWindow>,
}

/// Alert suppression rules
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Conditions for suppression
    pub conditions: Vec<SuppressionCondition>,

    /// Duration of suppression
    pub duration: Option<Duration>,

    /// Affected alert patterns
    pub alert_patterns: Vec<String>,

    /// Rule priority
    pub priority: u32,
}

/// Suppression conditions
#[derive(Debug, Clone)]
pub struct SuppressionCondition {
    /// Condition type
    pub condition_type: SuppressionConditionType,
    /// Condition parameters
    pub parameters: HashMap<String, String>,
}

/// Types of suppression conditions
#[derive(Debug, Clone)]
pub enum SuppressionConditionType {
    /// Time-based suppression
    Time { start: String, end: String, timezone: String },
    /// Service-based suppression
    Service { services: Vec<String> },
    /// Severity-based suppression
    Severity { severities: Vec<SeverityLevel> },
    /// Metric-based suppression
    Metric { metric: String, condition: AlertCondition },
    /// Custom suppression condition
    Custom { condition_name: String },
}

/// Global suppression settings
#[derive(Debug, Clone)]
pub struct GlobalSuppressionConfig {
    /// Enable global suppression
    pub enabled: bool,

    /// Default suppression duration
    pub default_duration: Duration,

    /// Maximum suppression duration
    pub max_duration: Duration,

    /// Require justification for suppression
    pub require_justification: bool,
}

/// Maintenance window configuration
#[derive(Debug, Clone)]
pub struct MaintenanceWindow {
    /// Window name
    pub name: String,

    /// Window description
    pub description: String,

    /// Start time
    pub start_time: std::time::SystemTime,

    /// End time
    pub end_time: std::time::SystemTime,

    /// Affected services/components
    pub affected_components: Vec<String>,

    /// Suppression behavior during window
    pub suppression_behavior: MaintenanceSuppressionBehavior,

    /// Recurrence pattern (for recurring maintenance)
    pub recurrence: Option<RecurrencePattern>,
}

/// Suppression behavior during maintenance windows
#[derive(Debug, Clone)]
pub enum MaintenanceSuppressionBehavior {
    /// Suppress all alerts
    SuppressAll,
    /// Suppress only specific alert types
    SuppressSpecific { alert_types: Vec<String> },
    /// Reduce alert severity
    ReduceSeverity { reduction_level: u8 },
    /// Route alerts to different channels
    RerouteAlerts { channels: Vec<String> },
}

/// Recurrence patterns for maintenance windows
#[derive(Debug, Clone)]
pub enum RecurrencePattern {
    /// Daily recurrence
    Daily,
    /// Weekly recurrence
    Weekly { day_of_week: u8 },
    /// Monthly recurrence
    Monthly { day_of_month: u8 },
    /// Custom cron pattern
    Cron { pattern: String },
}

/// Escalation configuration
///
/// Defines how alerts should escalate based on severity, time, and acknowledgment status.
#[derive(Debug, Clone)]
pub struct EscalationConfig {
    /// Enable escalation
    pub enabled: bool,

    /// Escalation policies
    pub policies: Vec<EscalationPolicy>,

    /// Default escalation settings
    pub default_settings: DefaultEscalationSettings,
}

/// Escalation policy definition
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// Policy name
    pub name: String,

    /// Policy description
    pub description: String,

    /// Escalation steps
    pub steps: Vec<EscalationStep>,

    /// Applicable severities
    pub severities: Vec<SeverityLevel>,

    /// Applicable services
    pub services: Vec<String>,
}

/// Individual escalation step
#[derive(Debug, Clone)]
pub struct EscalationStep {
    /// Step number (order)
    pub step_number: u32,

    /// Delay before this step
    pub delay: Duration,

    /// Target channels for this step
    pub channels: Vec<String>,

    /// Conditions for this step
    pub conditions: Vec<EscalationCondition>,

    /// Action to take
    pub action: EscalationAction,
}

/// Escalation conditions
#[derive(Debug, Clone)]
pub enum EscalationCondition {
    /// Time since alert creation
    TimeSinceCreation(Duration),
    /// Alert not acknowledged
    NotAcknowledged,
    /// Specific severity level
    SeverityLevel(SeverityLevel),
    /// Service-specific condition
    ServiceCondition { service: String, condition: String },
}

/// Escalation actions
#[derive(Debug, Clone)]
pub enum EscalationAction {
    /// Send notification
    Notify { channels: Vec<String> },
    /// Create incident
    CreateIncident { service: String },
    /// Execute webhook
    ExecuteWebhook { url: String, payload: HashMap<String, String> },
    /// Custom action
    Custom { action_name: String, parameters: HashMap<String, String> },
}

/// Default escalation settings
#[derive(Debug, Clone)]
pub struct DefaultEscalationSettings {
    /// Default escalation delay
    pub default_delay: Duration,

    /// Maximum escalation levels
    pub max_levels: u32,

    /// Auto-resolve after escalation
    pub auto_resolve: bool,

    /// Auto-resolve timeout
    pub auto_resolve_timeout: Duration,
}

/// Alert management configuration
///
/// Controls alert lifecycle management including acknowledgment, annotation, and resolution.
#[derive(Debug, Clone)]
pub struct AlertManagementConfig {
    /// Enable alert management
    pub enabled: bool,

    /// Acknowledgment settings
    pub acknowledgment: AcknowledgmentConfig,

    /// Auto-resolution settings
    pub auto_resolution: AutoResolutionConfig,

    /// Alert history retention
    pub history_retention: Duration,

    /// Alert annotation settings
    pub annotations: AnnotationConfig,
}

/// Acknowledgment configuration
#[derive(Debug, Clone)]
pub struct AcknowledgmentConfig {
    /// Enable acknowledgment
    pub enabled: bool,

    /// Acknowledgment timeout
    pub timeout: Duration,

    /// Require acknowledgment for specific severities
    pub required_severities: Vec<SeverityLevel>,

    /// Auto-acknowledge conditions
    pub auto_acknowledge: Vec<AutoAcknowledgeCondition>,
}

/// Auto-acknowledgment conditions
#[derive(Debug, Clone)]
pub enum AutoAcknowledgeCondition {
    /// After specific duration
    AfterDuration(Duration),
    /// When metric returns to normal
    MetricNormalized { metric: String, threshold: f64 },
    /// When dependent service recovers
    ServiceRecovered { service: String },
}

/// Auto-resolution configuration
#[derive(Debug, Clone)]
pub struct AutoResolutionConfig {
    /// Enable auto-resolution
    pub enabled: bool,

    /// Resolution timeout
    pub timeout: Duration,

    /// Resolution conditions
    pub conditions: Vec<AutoResolutionCondition>,
}

/// Auto-resolution conditions
#[derive(Debug, Clone)]
pub enum AutoResolutionCondition {
    /// Metric returns to normal range
    MetricNormal { metric: String, threshold: f64, duration: Duration },
    /// No new alerts of same type
    NoNewAlerts { duration: Duration },
    /// Manual resolution required
    ManualOnly,
}

/// Annotation configuration
#[derive(Debug, Clone)]
pub struct AnnotationConfig {
    /// Enable annotations
    pub enabled: bool,

    /// Maximum annotation length
    pub max_length: usize,

    /// Allow external annotations
    pub allow_external: bool,

    /// Annotation templates
    pub templates: Vec<AnnotationTemplate>,
}

/// Annotation template
#[derive(Debug, Clone)]
pub struct AnnotationTemplate {
    /// Template name
    pub name: String,

    /// Template content
    pub content: String,

    /// Template variables
    pub variables: Vec<String>,
}

impl AlertConfig {
    /// Create configuration optimized for production environments
    pub fn production() -> Self {
        Self {
            enabled: true,
            rules: vec![
                // High CPU utilization
                AlertRule {
                    name: "high_cpu_utilization".to_string(),
                    description: "CPU utilization is critically high".to_string(),
                    condition: AlertCondition::Threshold {
                        metric: "cpu_utilization".to_string(),
                        operator: ComparisonOperator::Greater,
                        value: 90.0,
                        duration: Duration::from_secs(300),
                        labels: HashMap::new(),
                    },
                    severity: SeverityLevel::Critical,
                    channels: vec!["pagerduty".to_string(), "slack".to_string()],
                    enabled: true,
                    evaluation_interval: Duration::from_secs(30),
                    metadata: HashMap::new(),
                    dependencies: Vec::new(),
                    cooldown: Some(Duration::from_secs(600)),
                },
                // High memory usage
                AlertRule {
                    name: "high_memory_usage".to_string(),
                    description: "Memory usage is critically high".to_string(),
                    condition: AlertCondition::Threshold {
                        metric: "memory_usage".to_string(),
                        operator: ComparisonOperator::Greater,
                        value: 95.0,
                        duration: Duration::from_secs(180),
                        labels: HashMap::new(),
                    },
                    severity: SeverityLevel::Critical,
                    channels: vec!["pagerduty".to_string(), "slack".to_string()],
                    enabled: true,
                    evaluation_interval: Duration::from_secs(30),
                    metadata: HashMap::new(),
                    dependencies: Vec::new(),
                    cooldown: Some(Duration::from_secs(300)),
                },
                // High error rate
                AlertRule {
                    name: "high_error_rate".to_string(),
                    description: "Error rate is above acceptable threshold".to_string(),
                    condition: AlertCondition::Rate {
                        metric: "error_rate".to_string(),
                        rate_threshold: 0.05, // 5% error rate
                        window: Duration::from_secs(300),
                        direction: RateDirection::Increase,
                    },
                    severity: SeverityLevel::Warning,
                    channels: vec!["slack".to_string(), "email".to_string()],
                    enabled: true,
                    evaluation_interval: Duration::from_secs(60),
                    metadata: HashMap::new(),
                    dependencies: Vec::new(),
                    cooldown: Some(Duration::from_secs(900)),
                },
            ],
            channels: vec![
                AlertChannel::PagerDuty {
                    name: "pagerduty".to_string(),
                    integration_key: "YOUR_PAGERDUTY_KEY".to_string(),
                    service: PagerDutyService {
                        name: "Production Monitoring".to_string(),
                        id: "PROD001".to_string(),
                        escalation_policy: "primary".to_string(),
                    },
                },
                AlertChannel::Slack {
                    name: "slack".to_string(),
                    webhook_url: "YOUR_SLACK_WEBHOOK".to_string(),
                    channel: "#alerts".to_string(),
                    formatting: SlackFormatting {
                        rich_formatting: true,
                        include_context: true,
                        colors: {
                            let mut colors = HashMap::new();
                            colors.insert("critical".to_string(), "#FF0000".to_string());
                            colors.insert("warning".to_string(), "#FFA500".to_string());
                            colors.insert("info".to_string(), "#0000FF".to_string());
                            colors
                        },
                    },
                },
                AlertChannel::Email {
                    name: "email".to_string(),
                    smtp_config: SmtpConfig {
                        host: "smtp.company.com".to_string(),
                        port: 587,
                        use_tls: true,
                        auth: Some(SmtpAuth {
                            username: "alerts@company.com".to_string(),
                            password: "password".to_string(),
                        }),
                        from: "alerts@company.com".to_string(),
                    },
                    recipients: vec!["oncall@company.com".to_string()],
                    template: EmailTemplate {
                        subject: "[{{severity}}] {{alert_name}}".to_string(),
                        body_html: "<h2>Alert: {{alert_name}}</h2><p>{{description}}</p>".to_string(),
                        body_text: "Alert: {{alert_name}}\n{{description}}".to_string(),
                        include_details: true,
                    },
                },
            ],
            aggregation: AlertAggregationConfig {
                enabled: true,
                window_size: Duration::from_secs(300),
                grouping_keys: vec!["service".to_string(), "severity".to_string()],
                max_alerts_per_group: 10,
                strategies: vec![AggregationStrategy::SeverityBased, AggregationStrategy::TimeBased],
                deduplication: DeduplicationConfig {
                    enabled: true,
                    dedup_keys: vec!["alert_name".to_string(), "instance".to_string()],
                    window: Duration::from_secs(600),
                    duplicate_action: DuplicateAction::Update,
                },
            },
            suppression: AlertSuppressionConfig {
                enabled: true,
                rules: Vec::new(),
                global: GlobalSuppressionConfig {
                    enabled: true,
                    default_duration: Duration::from_secs(3600),
                    max_duration: Duration::from_secs(86400),
                    require_justification: true,
                },
                maintenance_windows: Vec::new(),
            },
            escalation: EscalationConfig {
                enabled: true,
                policies: vec![EscalationPolicy {
                    name: "production_escalation".to_string(),
                    description: "Standard production escalation policy".to_string(),
                    steps: vec![
                        EscalationStep {
                            step_number: 1,
                            delay: Duration::from_secs(0),
                            channels: vec!["slack".to_string()],
                            conditions: vec![EscalationCondition::SeverityLevel(SeverityLevel::Warning)],
                            action: EscalationAction::Notify {
                                channels: vec!["slack".to_string()],
                            },
                        },
                        EscalationStep {
                            step_number: 2,
                            delay: Duration::from_secs(900),
                            channels: vec!["pagerduty".to_string()],
                            conditions: vec![EscalationCondition::NotAcknowledged],
                            action: EscalationAction::Notify {
                                channels: vec!["pagerduty".to_string()],
                            },
                        },
                    ],
                    severities: vec![SeverityLevel::Critical, SeverityLevel::Warning],
                    services: vec!["production".to_string()],
                }],
                default_settings: DefaultEscalationSettings {
                    default_delay: Duration::from_secs(900),
                    max_levels: 3,
                    auto_resolve: true,
                    auto_resolve_timeout: Duration::from_secs(3600),
                },
            },
            management: AlertManagementConfig {
                enabled: true,
                acknowledgment: AcknowledgmentConfig {
                    enabled: true,
                    timeout: Duration::from_secs(3600),
                    required_severities: vec![SeverityLevel::Critical],
                    auto_acknowledge: Vec::new(),
                },
                auto_resolution: AutoResolutionConfig {
                    enabled: true,
                    timeout: Duration::from_secs(86400),
                    conditions: vec![AutoResolutionCondition::MetricNormal {
                        metric: "health_check".to_string(),
                        threshold: 1.0,
                        duration: Duration::from_secs(300),
                    }],
                },
                history_retention: Duration::from_secs(86400 * 90), // 90 days
                annotations: AnnotationConfig {
                    enabled: true,
                    max_length: 1000,
                    allow_external: true,
                    templates: Vec::new(),
                },
            },
        }
    }

    /// Create configuration optimized for development environments
    pub fn development() -> Self {
        Self {
            enabled: false, // Typically disabled in development
            rules: vec![
                AlertRule {
                    name: "dev_error_spike".to_string(),
                    description: "Unusual error spike in development".to_string(),
                    condition: AlertCondition::Rate {
                        metric: "error_rate".to_string(),
                        rate_threshold: 0.1, // 10% error rate (higher tolerance)
                        window: Duration::from_secs(600),
                        direction: RateDirection::Increase,
                    },
                    severity: SeverityLevel::Info,
                    channels: vec!["slack".to_string()],
                    enabled: true,
                    evaluation_interval: Duration::from_secs(300),
                    metadata: HashMap::new(),
                    dependencies: Vec::new(),
                    cooldown: Some(Duration::from_secs(1800)),
                },
            ],
            channels: vec![AlertChannel::Slack {
                name: "slack".to_string(),
                webhook_url: "YOUR_DEV_SLACK_WEBHOOK".to_string(),
                channel: "#dev-alerts".to_string(),
                formatting: SlackFormatting {
                    rich_formatting: false,
                    include_context: false,
                    colors: HashMap::new(),
                },
            }],
            aggregation: AlertAggregationConfig {
                enabled: false,
                window_size: Duration::from_secs(600),
                grouping_keys: vec!["service".to_string()],
                max_alerts_per_group: 5,
                strategies: vec![AggregationStrategy::Count],
                deduplication: DeduplicationConfig {
                    enabled: false,
                    dedup_keys: vec!["alert_name".to_string()],
                    window: Duration::from_secs(1800),
                    duplicate_action: DuplicateAction::Discard,
                },
            },
            suppression: AlertSuppressionConfig {
                enabled: false,
                rules: Vec::new(),
                global: GlobalSuppressionConfig {
                    enabled: false,
                    default_duration: Duration::from_secs(1800),
                    max_duration: Duration::from_secs(3600),
                    require_justification: false,
                },
                maintenance_windows: Vec::new(),
            },
            escalation: EscalationConfig {
                enabled: false,
                policies: Vec::new(),
                default_settings: DefaultEscalationSettings {
                    default_delay: Duration::from_secs(3600),
                    max_levels: 1,
                    auto_resolve: true,
                    auto_resolve_timeout: Duration::from_secs(1800),
                },
            },
            management: AlertManagementConfig {
                enabled: false,
                acknowledgment: AcknowledgmentConfig {
                    enabled: false,
                    timeout: Duration::from_secs(1800),
                    required_severities: Vec::new(),
                    auto_acknowledge: Vec::new(),
                },
                auto_resolution: AutoResolutionConfig {
                    enabled: true,
                    timeout: Duration::from_secs(1800),
                    conditions: vec![AutoResolutionCondition::NoNewAlerts {
                        duration: Duration::from_secs(600),
                    }],
                },
                history_retention: Duration::from_secs(86400 * 7), // 7 days
                annotations: AnnotationConfig {
                    enabled: false,
                    max_length: 500,
                    allow_external: false,
                    templates: Vec::new(),
                },
            },
        }
    }

    /// Validate the alert configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate rule names are unique
        let mut rule_names = std::collections::HashSet::new();
        for rule in &self.rules {
            if !rule_names.insert(&rule.name) {
                return Err(format!("Duplicate alert rule name: {}", rule.name));
            }
        }

        // Validate channel names are unique
        let mut channel_names = std::collections::HashSet::new();
        for channel in &self.channels {
            let name = match channel {
                AlertChannel::Email { name, .. } => name,
                AlertChannel::Slack { name, .. } => name,
                AlertChannel::PagerDuty { name, .. } => name,
                AlertChannel::Teams { name, .. } => name,
                AlertChannel::Discord { name, .. } => name,
                AlertChannel::Sms { name, .. } => name,
                AlertChannel::Webhook { name, .. } => name,
                AlertChannel::Custom { name, .. } => name,
            };
            if !channel_names.insert(name) {
                return Err(format!("Duplicate alert channel name: {}", name));
            }
        }

        // Validate rule channel references
        let available_channels: std::collections::HashSet<&String> = channel_names.iter().collect();
        for rule in &self.rules {
            for channel_ref in &rule.channels {
                if !available_channels.contains(&channel_ref) {
                    return Err(format!(
                        "Rule '{}' references unknown channel '{}'",
                        rule.name, channel_ref
                    ));
                }
            }
        }

        // Validate aggregation settings
        if self.aggregation.enabled {
            if self.aggregation.max_alerts_per_group == 0 {
                return Err("Maximum alerts per group must be positive".to_string());
            }
        }

        Ok(())
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: Vec::new(),
            channels: Vec::new(),
            aggregation: AlertAggregationConfig {
                enabled: false,
                window_size: Duration::from_secs(300),
                grouping_keys: vec!["service".to_string()],
                max_alerts_per_group: 10,
                strategies: vec![AggregationStrategy::TimeBased],
                deduplication: DeduplicationConfig {
                    enabled: false,
                    dedup_keys: vec!["alert_name".to_string()],
                    window: Duration::from_secs(600),
                    duplicate_action: DuplicateAction::Update,
                },
            },
            suppression: AlertSuppressionConfig {
                enabled: false,
                rules: Vec::new(),
                global: GlobalSuppressionConfig {
                    enabled: false,
                    default_duration: Duration::from_secs(3600),
                    max_duration: Duration::from_secs(86400),
                    require_justification: false,
                },
                maintenance_windows: Vec::new(),
            },
            escalation: EscalationConfig {
                enabled: false,
                policies: Vec::new(),
                default_settings: DefaultEscalationSettings {
                    default_delay: Duration::from_secs(900),
                    max_levels: 2,
                    auto_resolve: false,
                    auto_resolve_timeout: Duration::from_secs(3600),
                },
            },
            management: AlertManagementConfig {
                enabled: false,
                acknowledgment: AcknowledgmentConfig {
                    enabled: false,
                    timeout: Duration::from_secs(3600),
                    required_severities: Vec::new(),
                    auto_acknowledge: Vec::new(),
                },
                auto_resolution: AutoResolutionConfig {
                    enabled: false,
                    timeout: Duration::from_secs(86400),
                    conditions: Vec::new(),
                },
                history_retention: Duration::from_secs(86400 * 30), // 30 days
                annotations: AnnotationConfig {
                    enabled: false,
                    max_length: 1000,
                    allow_external: false,
                    templates: Vec::new(),
                },
            },
        }
    }
}

impl AlertRule {
    /// Create a simple threshold-based alert rule
    pub fn threshold(name: &str, metric: &str, threshold: f64) -> Self {
        Self {
            name: name.to_string(),
            description: format!("Alert when {} exceeds {}", metric, threshold),
            condition: AlertCondition::Threshold {
                metric: metric.to_string(),
                operator: ComparisonOperator::Greater,
                value: threshold,
                duration: Duration::from_secs(300),
                labels: HashMap::new(),
            },
            severity: SeverityLevel::Warning,
            channels: Vec::new(),
            enabled: true,
            evaluation_interval: Duration::from_secs(60),
            metadata: HashMap::new(),
            dependencies: Vec::new(),
            cooldown: None,
        }
    }

    /// Create a rate-based alert rule
    pub fn rate(name: &str, metric: &str, rate_threshold: f64) -> Self {
        Self {
            name: name.to_string(),
            description: format!("Alert when {} rate exceeds {}", metric, rate_threshold),
            condition: AlertCondition::Rate {
                metric: metric.to_string(),
                rate_threshold,
                window: Duration::from_secs(300),
                direction: RateDirection::Increase,
            },
            severity: SeverityLevel::Warning,
            channels: Vec::new(),
            enabled: true,
            evaluation_interval: Duration::from_secs(60),
            metadata: HashMap::new(),
            dependencies: Vec::new(),
            cooldown: None,
        }
    }
}