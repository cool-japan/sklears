use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::dashboard_visualization::{
    WidgetType, ChartConfiguration, BorderConfiguration, ShadowConfiguration,
    PaddingConfiguration, FilterType, FilterOperator, InteractionTrigger,
    InteractionAction, InteractionTarget
};
use super::dashboard_datasources::ApiAuthentication;

/// Widget definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub widget_id: String,
    pub widget_type: WidgetType,
    pub title: String,
    pub description: Option<String>,
    pub position: WidgetPosition,
    pub size: WidgetSize,
    pub data_source: String,
    pub chart_config: ChartConfiguration,
    pub styling: WidgetStyling,
    pub filters: Vec<WidgetFilter>,
    pub interactions: Vec<WidgetInteraction>,
    pub alerts: Vec<WidgetAlert>,
    pub permissions: WidgetPermissions,
    pub metadata: WidgetMetadata,
}

/// Widget position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPosition {
    pub x: f64,
    pub y: f64,
    pub z_index: u32,
}

/// Widget size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSize {
    pub width: f64,
    pub height: f64,
    pub min_width: Option<f64>,
    pub min_height: Option<f64>,
    pub max_width: Option<f64>,
    pub max_height: Option<f64>,
    pub aspect_ratio: Option<f64>,
    pub resizable: bool,
}

/// Widget styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetStyling {
    pub background_color: String,
    pub border: BorderConfiguration,
    pub shadow: ShadowConfiguration,
    pub margin: PaddingConfiguration,
    pub padding: PaddingConfiguration,
    pub font_family: String,
    pub font_size: f64,
    pub text_color: String,
    pub theme_override: Option<String>,
}

/// Widget filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetFilter {
    pub filter_id: String,
    pub field: String,
    pub filter_type: FilterType,
    pub operator: FilterOperator,
    pub value: String,
    pub enabled: bool,
    pub user_configurable: bool,
}

/// Widget interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetInteraction {
    pub interaction_id: String,
    pub trigger: InteractionTrigger,
    pub action: InteractionAction,
    pub target: InteractionTarget,
    pub enabled: bool,
}

/// Widget alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetAlert {
    pub alert_id: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub notification: AlertNotification,
    pub enabled: bool,
}

/// Alert conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    ValueThreshold,
    PercentageChange,
    TrendChange,
    AnomalyDetection,
    DataFreshness,
    Custom(String),
}

/// Alert severities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertNotification {
    pub channels: Vec<NotificationChannel>,
    pub message_template: String,
    pub throttling: AlertThrottling,
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email(Vec<String>),
    SMS(Vec<String>),
    Slack(SlackConfig),
    Teams(TeamsConfig),
    Webhook(WebhookConfig),
    PagerDuty(PagerDutyConfig),
    Custom(String),
}

/// Slack configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    pub webhook_url: String,
    pub channel: String,
    pub username: Option<String>,
    pub emoji: Option<String>,
}

/// Teams configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamsConfig {
    pub webhook_url: String,
    pub title: Option<String>,
    pub theme_color: Option<String>,
}

/// Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    pub url: String,
    pub method: String,
    pub headers: HashMap<String, String>,
    pub authentication: Option<ApiAuthentication>,
}

/// PagerDuty configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagerDutyConfig {
    pub integration_key: String,
    pub severity: String,
    pub component: Option<String>,
    pub group: Option<String>,
    pub class: Option<String>,
}

/// Alert throttling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThrottling {
    pub enabled: bool,
    pub window: Duration,
    pub max_alerts: u32,
    pub escalation_rules: Vec<EscalationRule>,
}

/// Escalation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    pub after_duration: Duration,
    pub action: EscalationAction,
}

/// Escalation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationAction {
    IncreaseFrequency,
    AddNotificationChannel(NotificationChannel),
    ChangeMessageTemplate(String),
    TriggerRunbook(String),
    Custom(String),
}

/// Widget permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPermissions {
    pub view_permissions: Vec<String>,
    pub edit_permissions: Vec<String>,
    pub delete_permissions: Vec<String>,
    pub share_permissions: Vec<String>,
    pub export_permissions: Vec<String>,
}

/// Widget metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetMetadata {
    pub created_at: SystemTime,
    pub created_by: String,
    pub modified_at: SystemTime,
    pub modified_by: String,
    pub version: String,
    pub tags: Vec<String>,
    pub category: String,
    pub usage_statistics: UsageStatistics,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatistics {
    pub view_count: u64,
    pub interaction_count: u64,
    pub export_count: u64,
    pub last_accessed: Option<SystemTime>,
    pub popular_filters: Vec<String>,
    pub average_session_duration: Duration,
}

/// Widget builder for fluent API
#[derive(Debug, Clone)]
pub struct WidgetBuilder {
    widget: Widget,
}

impl WidgetBuilder {
    pub fn new(widget_id: String, widget_type: WidgetType) -> Self {
        Self {
            widget: Widget {
                widget_id,
                widget_type,
                title: String::new(),
                description: None,
                position: WidgetPosition { x: 0.0, y: 0.0, z_index: 0 },
                size: WidgetSize {
                    width: 300.0,
                    height: 200.0,
                    min_width: None,
                    min_height: None,
                    max_width: None,
                    max_height: None,
                    aspect_ratio: None,
                    resizable: true,
                },
                data_source: String::new(),
                chart_config: ChartConfiguration::default(),
                styling: WidgetStyling::default(),
                filters: Vec::new(),
                interactions: Vec::new(),
                alerts: Vec::new(),
                permissions: WidgetPermissions::default(),
                metadata: WidgetMetadata::default(),
            },
        }
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.widget.title = title.into();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.widget.description = Some(description.into());
        self
    }

    pub fn position(mut self, x: f64, y: f64) -> Self {
        self.widget.position.x = x;
        self.widget.position.y = y;
        self
    }

    pub fn size(mut self, width: f64, height: f64) -> Self {
        self.widget.size.width = width;
        self.widget.size.height = height;
        self
    }

    pub fn data_source(mut self, data_source: impl Into<String>) -> Self {
        self.widget.data_source = data_source.into();
        self
    }

    pub fn styling(mut self, styling: WidgetStyling) -> Self {
        self.widget.styling = styling;
        self
    }

    pub fn add_filter(mut self, filter: WidgetFilter) -> Self {
        self.widget.filters.push(filter);
        self
    }

    pub fn add_interaction(mut self, interaction: WidgetInteraction) -> Self {
        self.widget.interactions.push(interaction);
        self
    }

    pub fn add_alert(mut self, alert: WidgetAlert) -> Self {
        self.widget.alerts.push(alert);
        self
    }

    pub fn permissions(mut self, permissions: WidgetPermissions) -> Self {
        self.widget.permissions = permissions;
        self
    }

    pub fn build(self) -> Widget {
        self.widget
    }
}

impl Default for WidgetStyling {
    fn default() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            border: BorderConfiguration::default(),
            shadow: ShadowConfiguration::default(),
            margin: PaddingConfiguration::default(),
            padding: PaddingConfiguration::default(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: 14.0,
            text_color: "#000000".to_string(),
            theme_override: None,
        }
    }
}

impl Default for WidgetPermissions {
    fn default() -> Self {
        Self {
            view_permissions: vec!["*".to_string()],
            edit_permissions: vec!["admin".to_string()],
            delete_permissions: vec!["admin".to_string()],
            share_permissions: vec!["user".to_string()],
            export_permissions: vec!["user".to_string()],
        }
    }
}

impl Default for WidgetMetadata {
    fn default() -> Self {
        Self {
            created_at: SystemTime::now(),
            created_by: "system".to_string(),
            modified_at: SystemTime::now(),
            modified_by: "system".to_string(),
            version: "1.0.0".to_string(),
            tags: Vec::new(),
            category: "general".to_string(),
            usage_statistics: UsageStatistics::default(),
        }
    }
}

impl Default for UsageStatistics {
    fn default() -> Self {
        Self {
            view_count: 0,
            interaction_count: 0,
            export_count: 0,
            last_accessed: None,
            popular_filters: Vec::new(),
            average_session_duration: Duration::from_secs(0),
        }
    }
}

/// Widget validation
pub struct WidgetValidator;

impl WidgetValidator {
    pub fn validate(widget: &Widget) -> Result<(), WidgetValidationError> {
        if widget.widget_id.is_empty() {
            return Err(WidgetValidationError::EmptyWidgetId);
        }

        if widget.title.is_empty() {
            return Err(WidgetValidationError::EmptyTitle);
        }

        if widget.data_source.is_empty() {
            return Err(WidgetValidationError::EmptyDataSource);
        }

        if widget.size.width <= 0.0 || widget.size.height <= 0.0 {
            return Err(WidgetValidationError::InvalidSize);
        }

        // Validate min/max constraints
        if let (Some(min_width), Some(max_width)) = (widget.size.min_width, widget.size.max_width) {
            if min_width > max_width {
                return Err(WidgetValidationError::InvalidSizeConstraints);
            }
        }

        if let (Some(min_height), Some(max_height)) = (widget.size.min_height, widget.size.max_height) {
            if min_height > max_height {
                return Err(WidgetValidationError::InvalidSizeConstraints);
            }
        }

        // Validate alerts
        for alert in &widget.alerts {
            if alert.alert_id.is_empty() {
                return Err(WidgetValidationError::InvalidAlert("Empty alert ID".to_string()));
            }

            if alert.notification.channels.is_empty() {
                return Err(WidgetValidationError::InvalidAlert("No notification channels".to_string()));
            }
        }

        // Validate interactions
        for interaction in &widget.interactions {
            if interaction.interaction_id.is_empty() {
                return Err(WidgetValidationError::InvalidInteraction("Empty interaction ID".to_string()));
            }
        }

        Ok(())
    }
}

/// Widget validation errors
#[derive(Debug, Clone)]
pub enum WidgetValidationError {
    EmptyWidgetId,
    EmptyTitle,
    EmptyDataSource,
    InvalidSize,
    InvalidSizeConstraints,
    InvalidAlert(String),
    InvalidInteraction(String),
}

impl std::fmt::Display for WidgetValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyWidgetId => write!(f, "Widget ID cannot be empty"),
            Self::EmptyTitle => write!(f, "Widget title cannot be empty"),
            Self::EmptyDataSource => write!(f, "Widget data source cannot be empty"),
            Self::InvalidSize => write!(f, "Widget size must be positive"),
            Self::InvalidSizeConstraints => write!(f, "Widget size constraints are invalid"),
            Self::InvalidAlert(msg) => write!(f, "Invalid alert configuration: {}", msg),
            Self::InvalidInteraction(msg) => write!(f, "Invalid interaction configuration: {}", msg),
        }
    }
}

impl std::error::Error for WidgetValidationError {}

/// Widget collection manager
#[derive(Debug, Clone)]
pub struct WidgetCollection {
    pub widgets: HashMap<String, Widget>,
}

impl WidgetCollection {
    pub fn new() -> Self {
        Self {
            widgets: HashMap::new(),
        }
    }

    pub fn add_widget(&mut self, widget: Widget) -> Result<(), WidgetValidationError> {
        WidgetValidator::validate(&widget)?;
        self.widgets.insert(widget.widget_id.clone(), widget);
        Ok(())
    }

    pub fn remove_widget(&mut self, widget_id: &str) -> Option<Widget> {
        self.widgets.remove(widget_id)
    }

    pub fn get_widget(&self, widget_id: &str) -> Option<&Widget> {
        self.widgets.get(widget_id)
    }

    pub fn get_widget_mut(&mut self, widget_id: &str) -> Option<&mut Widget> {
        self.widgets.get_mut(widget_id)
    }

    pub fn update_widget(&mut self, widget: Widget) -> Result<(), WidgetValidationError> {
        WidgetValidator::validate(&widget)?;
        self.widgets.insert(widget.widget_id.clone(), widget);
        Ok(())
    }

    pub fn list_widgets(&self) -> Vec<&Widget> {
        self.widgets.values().collect()
    }

    pub fn filter_widgets_by_type(&self, widget_type: &WidgetType) -> Vec<&Widget> {
        self.widgets
            .values()
            .filter(|widget| &widget.widget_type == widget_type)
            .collect()
    }

    pub fn filter_widgets_by_category(&self, category: &str) -> Vec<&Widget> {
        self.widgets
            .values()
            .filter(|widget| widget.metadata.category == category)
            .collect()
    }

    pub fn get_widget_count(&self) -> usize {
        self.widgets.len()
    }

    pub fn clear(&mut self) {
        self.widgets.clear();
    }
}

impl Default for WidgetCollection {
    fn default() -> Self {
        Self::new()
    }
}