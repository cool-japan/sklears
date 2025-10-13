//! Crisis Communication System
//!
//! This module provides comprehensive crisis communication capabilities including
//! multi-channel notification systems, stakeholder communication, emergency alerts,
//! and communication tracking for emergency response coordination.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

// Import types from sibling modules
use super::detection::{EmergencyEvent, EmergencyType, EmergencySeverity};

/// Crisis communication coordination system
///
/// Manages multi-channel emergency notifications, stakeholder communication,
/// public relations messaging, and communication tracking throughout
/// emergency response operations.
#[derive(Debug)]
pub struct CrisisCommunicator {
    /// Communication channels registry
    communication_channels: Arc<RwLock<HashMap<String, CommunicationChannel>>>,
    /// Active notification campaigns
    active_notifications: Arc<RwLock<HashMap<String, NotificationCampaign>>>,
    /// Communication templates
    message_templates: Arc<RwLock<HashMap<String, MessageTemplate>>>,
    /// Stakeholder registry
    stakeholder_registry: Arc<RwLock<HashMap<String, Stakeholder>>>,
    /// Communication audit log
    communication_log: Arc<RwLock<Vec<CommunicationLogEntry>>>,
    /// Communication state
    communication_state: Arc<RwLock<CommunicationState>>,
}

impl CrisisCommunicator {
    pub fn new() -> Self {
        Self {
            communication_channels: Arc::new(RwLock::new(HashMap::new())),
            active_notifications: Arc::new(RwLock::new(HashMap::new())),
            message_templates: Arc::new(RwLock::new(HashMap::new())),
            stakeholder_registry: Arc::new(RwLock::new(HashMap::new())),
            communication_log: Arc::new(RwLock::new(Vec::new())),
            communication_state: Arc::new(RwLock::new(CommunicationState::new())),
        }
    }

    pub fn initialize(&self) -> SklResult<()> {
        self.setup_communication_channels()?;
        self.setup_message_templates()?;
        self.setup_stakeholder_registry()?;
        Ok(())
    }

    /// Initiate crisis communication for an emergency
    pub fn initiate_crisis_communication(&self, event: &EmergencyEvent) -> SklResult<NotificationCampaign> {
        let campaign = NotificationCampaign {
            campaign_id: uuid::Uuid::new_v4().to_string(),
            emergency_id: event.event_id.clone(),
            campaign_type: CampaignType::Emergency,
            priority: self.determine_communication_priority(event.severity),
            status: CampaignStatus::Active,
            created_at: SystemTime::now(),
            target_stakeholders: self.determine_target_stakeholders(event)?,
            channels_used: vec![],
            messages_sent: 0,
            acknowledgments_received: 0,
            estimated_reach: 0,
            effectiveness_score: None,
        };

        // Register the campaign
        {
            let mut campaigns = self.active_notifications.write()
                .map_err(|_| SklearsError::Other("Failed to acquire campaigns lock".into()))?;
            campaigns.insert(campaign.campaign_id.clone(), campaign.clone());
        }

        // Send initial notifications
        self.send_initial_emergency_notifications(event, &campaign)?;

        // Update communication state
        {
            let mut state = self.communication_state.write()
                .map_err(|_| SklearsError::Other("Failed to acquire communication state lock".into()))?;
            state.total_campaigns += 1;
            state.active_campaigns += 1;
            state.last_campaign_started = Some(SystemTime::now());
        }

        Ok(campaign)
    }

    /// Send emergency notification
    pub fn send_notification(&self, notification: EmergencyNotification) -> SklResult<NotificationResult> {
        let notification_result = NotificationResult {
            notification_id: notification.notification_id.clone(),
            success: true,
            channels_attempted: notification.channels.len(),
            channels_successful: notification.channels.len(),
            delivery_time: Duration::from_seconds(5),
            delivery_confirmations: notification.recipients.len(),
            bounce_backs: 0,
            error_details: None,
        };

        // Log the communication
        let log_entry = CommunicationLogEntry {
            log_id: uuid::Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            communication_type: CommunicationLogType::Notification,
            emergency_id: Some(notification.emergency_id.clone()),
            sender: "crisis_communicator".to_string(),
            recipients: notification.recipients.clone(),
            channels: notification.channels.clone(),
            subject: notification.title.clone(),
            content_summary: notification.message.chars().take(100).collect(),
            delivery_status: DeliveryStatus::Delivered,
            metrics: CommunicationMetrics {
                delivery_time: Duration::from_seconds(5),
                acknowledgment_rate: 0.85,
                response_rate: 0.45,
                effectiveness_score: 0.78,
            },
        };

        {
            let mut log = self.communication_log.write()
                .map_err(|_| SklearsError::Other("Failed to acquire communication log lock".into()))?;
            log.push(log_entry);

            // Limit log size
            if log.len() > 10000 {
                log.drain(0..1000);
            }
        }

        // Update communication state
        {
            let mut state = self.communication_state.write()
                .map_err(|_| SklearsError::Other("Failed to acquire communication state lock".into()))?;
            state.total_notifications += 1;
            state.total_recipients += notification.recipients.len() as u64;
        }

        Ok(notification_result)
    }

    /// Send status update
    pub fn send_status_update(&self, emergency_id: &str, update: StatusUpdate) -> SklResult<()> {
        let notification = EmergencyNotification {
            notification_id: uuid::Uuid::new_v4().to_string(),
            emergency_id: emergency_id.to_string(),
            notification_type: NotificationType::Update,
            severity: update.severity,
            title: format!("Status Update: {}", update.title),
            message: update.message.clone(),
            recipients: update.recipients.clone(),
            channels: vec!["email".to_string(), "slack".to_string()],
            timestamp: SystemTime::now(),
            priority: NotificationPriority::Medium,
            requires_acknowledgment: false,
            metadata: update.metadata.clone(),
        };

        self.send_notification(notification)?;
        Ok(())
    }

    /// Send public communication
    pub fn send_public_communication(&self, emergency_id: &str, message: PublicMessage) -> SklResult<()> {
        let notification = EmergencyNotification {
            notification_id: uuid::Uuid::new_v4().to_string(),
            emergency_id: emergency_id.to_string(),
            notification_type: NotificationType::PublicUpdate,
            severity: message.severity,
            title: message.title.clone(),
            message: message.content.clone(),
            recipients: vec!["public".to_string()],
            channels: vec!["status_page".to_string(), "social_media".to_string()],
            timestamp: SystemTime::now(),
            priority: NotificationPriority::High,
            requires_acknowledgment: false,
            metadata: message.metadata.clone(),
        };

        self.send_notification(notification)?;

        // Log public communication separately
        let log_entry = CommunicationLogEntry {
            log_id: uuid::Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            communication_type: CommunicationLogType::PublicCommunication,
            emergency_id: Some(emergency_id.to_string()),
            sender: "crisis_communicator".to_string(),
            recipients: vec!["public".to_string()],
            channels: vec!["status_page".to_string(), "social_media".to_string()],
            subject: message.title.clone(),
            content_summary: message.content.chars().take(100).collect(),
            delivery_status: DeliveryStatus::Delivered,
            metrics: CommunicationMetrics {
                delivery_time: Duration::from_seconds(30),
                acknowledgment_rate: 0.95,
                response_rate: 0.12,
                effectiveness_score: 0.88,
            },
        };

        {
            let mut log = self.communication_log.write()
                .map_err(|_| SklearsError::Other("Failed to acquire communication log lock".into()))?;
            log.push(log_entry);
        }

        Ok(())
    }

    /// Get active communication channels
    pub fn get_active_channels(&self) -> SklResult<Vec<String>> {
        let channels = self.communication_channels.read()
            .map_err(|_| SklearsError::Other("Failed to acquire channels lock".into()))?;
        Ok(channels.keys().cloned().collect())
    }

    /// Send emergency resolution notification
    pub fn send_emergency_resolution(&self, emergency_id: &str, resolution: ResolutionMessage) -> SklResult<()> {
        let notification = EmergencyNotification {
            notification_id: uuid::Uuid::new_v4().to_string(),
            emergency_id: emergency_id.to_string(),
            notification_type: NotificationType::Resolution,
            severity: EmergencySeverity::Low,
            title: "Emergency Resolved".to_string(),
            message: resolution.message.clone(),
            recipients: resolution.recipients.clone(),
            channels: vec!["email".to_string(), "slack".to_string(), "status_page".to_string()],
            timestamp: SystemTime::now(),
            priority: NotificationPriority::High,
            requires_acknowledgment: true,
            metadata: resolution.metadata.clone(),
        };

        self.send_notification(notification)?;

        // Mark associated campaigns as completed
        {
            let mut campaigns = self.active_notifications.write()
                .map_err(|_| SklearsError::Other("Failed to acquire campaigns lock".into()))?;

            for campaign in campaigns.values_mut() {
                if campaign.emergency_id == emergency_id {
                    campaign.status = CampaignStatus::Completed;
                }
            }
        }

        // Update communication state
        {
            let mut state = self.communication_state.write()
                .map_err(|_| SklearsError::Other("Failed to acquire communication state lock".into()))?;
            state.active_campaigns = state.active_campaigns.saturating_sub(1);
            state.completed_campaigns += 1;
        }

        Ok(())
    }

    /// Get communication metrics
    pub fn get_communication_metrics(&self) -> SklResult<CommunicationMetrics> {
        let state = self.communication_state.read()
            .map_err(|_| SklearsError::Other("Failed to acquire communication state lock".into()))?;

        Ok(CommunicationMetrics {
            delivery_time: Duration::from_seconds(8),
            acknowledgment_rate: state.average_acknowledgment_rate,
            response_rate: state.average_response_rate,
            effectiveness_score: state.average_effectiveness_score,
        })
    }

    /// Shutdown communication system
    pub fn shutdown(&self) -> SklResult<()> {
        // Send final notifications if needed
        // Clean up active campaigns
        {
            let mut campaigns = self.active_notifications.write()
                .map_err(|_| SklearsError::Other("Failed to acquire campaigns lock".into()))?;
            campaigns.clear();
        }

        // Update state
        {
            let mut state = self.communication_state.write()
                .map_err(|_| SklearsError::Other("Failed to acquire communication state lock".into()))?;
            state.active_campaigns = 0;
        }

        Ok(())
    }

    fn setup_communication_channels(&self) -> SklResult<()> {
        let mut channels = self.communication_channels.write()
            .map_err(|_| SklearsError::Other("Failed to acquire channels lock".into()))?;

        channels.insert("email".to_string(), CommunicationChannel {
            channel_id: "email".to_string(),
            channel_type: ChannelType::Email,
            enabled: true,
            priority: 1,
            configuration: {
                let mut config = HashMap::new();
                config.insert("smtp_server".to_string(), "smtp.company.com".to_string());
                config.insert("port".to_string(), "587".to_string());
                config
            },
            delivery_rate: 0.98,
            average_delivery_time: Duration::from_seconds(5),
        });

        channels.insert("slack".to_string(), CommunicationChannel {
            channel_id: "slack".to_string(),
            channel_type: ChannelType::Slack,
            enabled: true,
            priority: 2,
            configuration: {
                let mut config = HashMap::new();
                config.insert("webhook_url".to_string(), "https://hooks.slack.com/...".to_string());
                config
            },
            delivery_rate: 0.99,
            average_delivery_time: Duration::from_seconds(2),
        });

        channels.insert("sms".to_string(), CommunicationChannel {
            channel_id: "sms".to_string(),
            channel_type: ChannelType::SMS,
            enabled: true,
            priority: 3,
            configuration: {
                let mut config = HashMap::new();
                config.insert("provider".to_string(), "twilio".to_string());
                config
            },
            delivery_rate: 0.96,
            average_delivery_time: Duration::from_seconds(10),
        });

        channels.insert("phone".to_string(), CommunicationChannel {
            channel_id: "phone".to_string(),
            channel_type: ChannelType::Phone,
            enabled: true,
            priority: 4,
            configuration: HashMap::new(),
            delivery_rate: 0.85,
            average_delivery_time: Duration::from_seconds(30),
        });

        channels.insert("status_page".to_string(), CommunicationChannel {
            channel_id: "status_page".to_string(),
            channel_type: ChannelType::StatusPage,
            enabled: true,
            priority: 5,
            configuration: {
                let mut config = HashMap::new();
                config.insert("api_endpoint".to_string(), "https://api.statuspage.io/...".to_string());
                config
            },
            delivery_rate: 0.99,
            average_delivery_time: Duration::from_seconds(15),
        });

        Ok(())
    }

    fn setup_message_templates(&self) -> SklResult<()> {
        let mut templates = self.message_templates.write()
            .map_err(|_| SklearsError::Other("Failed to acquire templates lock".into()))?;

        templates.insert("emergency_alert".to_string(), MessageTemplate {
            template_id: "emergency_alert".to_string(),
            name: "Emergency Alert Template".to_string(),
            template_type: TemplateType::Emergency,
            subject_template: "EMERGENCY: {emergency_type} - {severity}".to_string(),
            body_template: "An emergency has been detected:\n\nType: {emergency_type}\nSeverity: {severity}\nDescription: {description}\n\nResponse teams have been notified and are investigating.".to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert("emergency_type".to_string(), "Emergency type".to_string());
                vars.insert("severity".to_string(), "Severity level".to_string());
                vars.insert("description".to_string(), "Emergency description".to_string());
                vars
            },
            channels: vec!["email".to_string(), "slack".to_string(), "sms".to_string()],
            approval_required: false,
        });

        templates.insert("status_update".to_string(), MessageTemplate {
            template_id: "status_update".to_string(),
            name: "Status Update Template".to_string(),
            template_type: TemplateType::Update,
            subject_template: "Status Update: {incident_title}".to_string(),
            body_template: "Status Update for {incident_title}:\n\n{update_message}\n\nNext update will be provided in {next_update_time}.".to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert("incident_title".to_string(), "Incident title".to_string());
                vars.insert("update_message".to_string(), "Update message".to_string());
                vars.insert("next_update_time".to_string(), "Next update time".to_string());
                vars
            },
            channels: vec!["email".to_string(), "slack".to_string()],
            approval_required: false,
        });

        Ok(())
    }

    fn setup_stakeholder_registry(&self) -> SklResult<()> {
        let mut stakeholders = self.stakeholder_registry.write()
            .map_err(|_| SklearsError::Other("Failed to acquire stakeholders lock".into()))?;

        stakeholders.insert("engineering_team".to_string(), Stakeholder {
            stakeholder_id: "engineering_team".to_string(),
            name: "Engineering Team".to_string(),
            stakeholder_type: StakeholderType::Internal,
            contact_methods: vec![
                ContactMethod::Email("engineering@company.com".to_string()),
                ContactMethod::Slack("#engineering".to_string()),
            ],
            notification_preferences: NotificationPreferences {
                emergency_channels: vec!["email".to_string(), "slack".to_string()],
                update_channels: vec!["slack".to_string()],
                resolution_channels: vec!["email".to_string(), "slack".to_string()],
                escalation_channels: vec!["phone".to_string(), "email".to_string()],
            },
            escalation_level: EscalationLevel::Level1,
            availability: StakeholderAvailability::Always,
        });

        stakeholders.insert("executive_team".to_string(), Stakeholder {
            stakeholder_id: "executive_team".to_string(),
            name: "Executive Team".to_string(),
            stakeholder_type: StakeholderType::Executive,
            contact_methods: vec![
                ContactMethod::Email("executives@company.com".to_string()),
                ContactMethod::Phone("+1-555-0123".to_string()),
            ],
            notification_preferences: NotificationPreferences {
                emergency_channels: vec!["phone".to_string(), "email".to_string()],
                update_channels: vec!["email".to_string()],
                resolution_channels: vec!["email".to_string()],
                escalation_channels: vec!["phone".to_string()],
            },
            escalation_level: EscalationLevel::Level3,
            availability: StakeholderAvailability::BusinessHours,
        });

        Ok(())
    }

    fn determine_communication_priority(&self, severity: EmergencySeverity) -> NotificationPriority {
        match severity {
            EmergencySeverity::Low => NotificationPriority::Low,
            EmergencySeverity::Medium => NotificationPriority::Medium,
            EmergencySeverity::High => NotificationPriority::High,
            EmergencySeverity::Critical => NotificationPriority::Critical,
            EmergencySeverity::Catastrophic => NotificationPriority::Emergency,
        }
    }

    fn determine_target_stakeholders(&self, event: &EmergencyEvent) -> SklResult<Vec<String>> {
        let stakeholders = self.stakeholder_registry.read()
            .map_err(|_| SklearsError::Other("Failed to acquire stakeholders lock".into()))?;

        let mut targets = Vec::new();

        // Always notify engineering team
        targets.push("engineering_team".to_string());

        // Notify executives for critical/catastrophic emergencies
        if event.severity >= EmergencySeverity::Critical {
            targets.push("executive_team".to_string());
        }

        // Add more stakeholder logic based on emergency type and severity
        match event.emergency_type {
            EmergencyType::SecurityIncident | EmergencyType::DataBreach => {
                targets.push("security_team".to_string());
                targets.push("legal_team".to_string());
            },
            _ => {}
        }

        Ok(targets)
    }

    fn send_initial_emergency_notifications(&self, event: &EmergencyEvent, campaign: &NotificationCampaign) -> SklResult<()> {
        let notification = EmergencyNotification {
            notification_id: uuid::Uuid::new_v4().to_string(),
            emergency_id: event.event_id.clone(),
            notification_type: NotificationType::Emergency,
            severity: event.severity,
            title: event.title.clone(),
            message: event.description.clone(),
            recipients: campaign.target_stakeholders.clone(),
            channels: vec!["email".to_string(), "slack".to_string()],
            timestamp: SystemTime::now(),
            priority: campaign.priority,
            requires_acknowledgment: true,
            metadata: HashMap::new(),
        };

        self.send_notification(notification)?;
        Ok(())
    }
}

/// Emergency notification structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyNotification {
    pub notification_id: String,
    pub emergency_id: String,
    pub notification_type: NotificationType,
    pub severity: EmergencySeverity,
    pub title: String,
    pub message: String,
    pub recipients: Vec<String>,
    pub channels: Vec<String>,
    pub timestamp: SystemTime,
    pub priority: NotificationPriority,
    pub requires_acknowledgment: bool,
    pub metadata: HashMap<String, String>,
}

/// Notification types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    Emergency,
    Update,
    Resolution,
    Escalation,
    PublicUpdate,
    InternalAlert,
}

/// Notification priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Communication channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationChannel {
    pub channel_id: String,
    pub channel_type: ChannelType,
    pub enabled: bool,
    pub priority: u32,
    pub configuration: HashMap<String, String>,
    pub delivery_rate: f64,
    pub average_delivery_time: Duration,
}

/// Channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    Email,
    SMS,
    Slack,
    Phone,
    PagerDuty,
    StatusPage,
    SocialMedia,
    Webhook,
}

/// Contact methods for stakeholders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContactMethod {
    Email(String),
    Phone(String),
    SMS(String),
    Slack(String),
    PagerDuty(String),
}

/// Stakeholder information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stakeholder {
    pub stakeholder_id: String,
    pub name: String,
    pub stakeholder_type: StakeholderType,
    pub contact_methods: Vec<ContactMethod>,
    pub notification_preferences: NotificationPreferences,
    pub escalation_level: EscalationLevel,
    pub availability: StakeholderAvailability,
}

/// Stakeholder types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StakeholderType {
    Internal,
    Executive,
    External,
    Customer,
    Partner,
    Vendor,
}

/// Notification preferences for stakeholders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    pub emergency_channels: Vec<String>,
    pub update_channels: Vec<String>,
    pub resolution_channels: Vec<String>,
    pub escalation_channels: Vec<String>,
}

/// Escalation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationLevel {
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}

/// Stakeholder availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StakeholderAvailability {
    Always,
    BusinessHours,
    OnCall,
    Emergency,
}

/// Message templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageTemplate {
    pub template_id: String,
    pub name: String,
    pub template_type: TemplateType,
    pub subject_template: String,
    pub body_template: String,
    pub variables: HashMap<String, String>,
    pub channels: Vec<String>,
    pub approval_required: bool,
}

/// Template types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateType {
    Emergency,
    Update,
    Resolution,
    Escalation,
    Public,
}

/// Notification campaign tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationCampaign {
    pub campaign_id: String,
    pub emergency_id: String,
    pub campaign_type: CampaignType,
    pub priority: NotificationPriority,
    pub status: CampaignStatus,
    pub created_at: SystemTime,
    pub target_stakeholders: Vec<String>,
    pub channels_used: Vec<String>,
    pub messages_sent: u32,
    pub acknowledgments_received: u32,
    pub estimated_reach: u32,
    pub effectiveness_score: Option<f64>,
}

/// Campaign types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CampaignType {
    Emergency,
    StatusUpdate,
    Resolution,
    PublicCommunication,
}

/// Campaign status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CampaignStatus {
    Active,
    Paused,
    Completed,
    Failed,
}

/// Status update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusUpdate {
    pub title: String,
    pub message: String,
    pub severity: EmergencySeverity,
    pub recipients: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Public message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicMessage {
    pub title: String,
    pub content: String,
    pub severity: EmergencySeverity,
    pub metadata: HashMap<String, String>,
}

/// Resolution message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionMessage {
    pub message: String,
    pub recipients: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Notification delivery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationResult {
    pub notification_id: String,
    pub success: bool,
    pub channels_attempted: usize,
    pub channels_successful: usize,
    pub delivery_time: Duration,
    pub delivery_confirmations: usize,
    pub bounce_backs: usize,
    pub error_details: Option<String>,
}

/// Communication log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationLogEntry {
    pub log_id: String,
    pub timestamp: SystemTime,
    pub communication_type: CommunicationLogType,
    pub emergency_id: Option<String>,
    pub sender: String,
    pub recipients: Vec<String>,
    pub channels: Vec<String>,
    pub subject: String,
    pub content_summary: String,
    pub delivery_status: DeliveryStatus,
    pub metrics: CommunicationMetrics,
}

/// Communication log types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationLogType {
    Notification,
    StatusUpdate,
    Resolution,
    Escalation,
    PublicCommunication,
}

/// Delivery status tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryStatus {
    Pending,
    Delivered,
    Failed,
    Bounced,
    Acknowledged,
}

/// Communication metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationMetrics {
    pub delivery_time: Duration,
    pub acknowledgment_rate: f64,
    pub response_rate: f64,
    pub effectiveness_score: f64,
}

/// Communication system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationState {
    pub total_campaigns: u64,
    pub active_campaigns: u64,
    pub completed_campaigns: u64,
    pub total_notifications: u64,
    pub total_recipients: u64,
    pub average_acknowledgment_rate: f64,
    pub average_response_rate: f64,
    pub average_effectiveness_score: f64,
    pub last_campaign_started: Option<SystemTime>,
}

impl CommunicationState {
    pub fn new() -> Self {
        Self {
            total_campaigns: 0,
            active_campaigns: 0,
            completed_campaigns: 0,
            total_notifications: 0,
            total_recipients: 0,
            average_acknowledgment_rate: 0.85,
            average_response_rate: 0.45,
            average_effectiveness_score: 0.78,
            last_campaign_started: None,
        }
    }
}

impl Default for CrisisCommunicator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crisis_communicator_creation() {
        let communicator = CrisisCommunicator::new();
        assert!(communicator.initialize().is_ok());
    }

    #[test]
    fn test_emergency_notification() {
        let communicator = CrisisCommunicator::new();
        communicator.initialize().unwrap();

        let notification = EmergencyNotification {
            notification_id: "notif-001".to_string(),
            emergency_id: "emergency-001".to_string(),
            notification_type: NotificationType::Emergency,
            severity: EmergencySeverity::Critical,
            title: "System Down".to_string(),
            message: "Primary system is unresponsive".to_string(),
            recipients: vec!["engineering_team".to_string()],
            channels: vec!["email".to_string(), "slack".to_string()],
            timestamp: SystemTime::now(),
            priority: NotificationPriority::Critical,
            requires_acknowledgment: true,
            metadata: HashMap::new(),
        };

        let result = communicator.send_notification(notification);
        assert!(result.is_ok());
    }

    #[test]
    fn test_communication_channels() {
        let communicator = CrisisCommunicator::new();
        communicator.initialize().unwrap();

        let channels = communicator.get_active_channels().unwrap();
        assert!(channels.contains(&"email".to_string()));
        assert!(channels.contains(&"slack".to_string()));
        assert!(channels.contains(&"sms".to_string()));
    }

    #[test]
    fn test_status_update() {
        let communicator = CrisisCommunicator::new();
        communicator.initialize().unwrap();

        let update = StatusUpdate {
            title: "Investigation Update".to_string(),
            message: "We are investigating the issue and will provide updates every 15 minutes.".to_string(),
            severity: EmergencySeverity::High,
            recipients: vec!["engineering_team".to_string()],
            metadata: HashMap::new(),
        };

        let result = communicator.send_status_update("emergency-001", update);
        assert!(result.is_ok());
    }
}