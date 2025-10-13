//! Incident Management System
//!
//! This module provides comprehensive incident command and coordination capabilities
//! including incident creation, tracking, command assignment, response team coordination,
//! and incident lifecycle management.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

// Import types from sibling modules
use super::detection::{EmergencyEvent, EmergencyType, EmergencySeverity};

/// Incident command and coordination system
///
/// Manages incident creation, tracking, command assignment, and coordination
/// with response teams. Provides the central command structure for emergency
/// incident management with full lifecycle tracking.
#[derive(Debug)]
pub struct IncidentCommander {
    /// Active incidents registry
    active_incidents: Arc<RwLock<HashMap<String, Incident>>>,
    /// Incident history for analysis
    incident_history: Arc<RwLock<Vec<HistoricalIncident>>>,
    /// Command assignments
    command_assignments: Arc<RwLock<HashMap<String, CommandAssignment>>>,
    /// Incident coordination state
    coordination_state: Arc<RwLock<IncidentCoordinationState>>,
}

impl IncidentCommander {
    pub fn new() -> Self {
        Self {
            active_incidents: Arc::new(RwLock::new(HashMap::new())),
            incident_history: Arc::new(RwLock::new(Vec::new())),
            command_assignments: Arc::new(RwLock::new(HashMap::new())),
            coordination_state: Arc::new(RwLock::new(IncidentCoordinationState::new())),
        }
    }

    pub fn initialize(&self) -> SklResult<()> {
        self.setup_default_command_structure()?;
        Ok(())
    }

    /// Create a new incident from an emergency event
    pub fn create_incident(&self, event: &EmergencyEvent) -> SklResult<Incident> {
        let incident = Incident {
            incident_id: uuid::Uuid::new_v4().to_string(),
            emergency_event_id: event.event_id.clone(),
            title: event.title.clone(),
            description: event.description.clone(),
            severity: event.severity,
            status: IncidentStatus::Active,
            created_at: SystemTime::now(),
            commander_assigned: Some("incident_commander".to_string()),
            response_team: vec!["primary_responder".to_string()],
            estimated_resolution: event.estimated_impact_duration,
            actual_resolution_time: None,
            root_cause: None,
            lessons_learned: vec![],
            timeline: vec![
                IncidentTimelineEntry {
                    timestamp: SystemTime::now(),
                    event_type: TimelineEventType::IncidentCreated,
                    description: "Incident created from emergency event".to_string(),
                    actor: "incident_commander".to_string(),
                    details: HashMap::new(),
                }
            ],
            impact_assessment: IncidentImpact {
                affected_users: 0,
                affected_systems: event.affected_systems.clone(),
                business_impact: BusinessImpactLevel::High,
                estimated_revenue_loss: None,
                sla_violations: vec![],
            },
            communication_log: vec![],
            escalation_history: vec![],
            resolution_actions: vec![],
        };

        // Register the incident
        {
            let mut incidents = self.active_incidents.write()
                .map_err(|_| SklearsError::Other("Failed to acquire incidents lock".into()))?;
            incidents.insert(incident.incident_id.clone(), incident.clone());
        }

        // Update coordination state
        {
            let mut state = self.coordination_state.write()
                .map_err(|_| SklearsError::Other("Failed to acquire coordination state lock".into()))?;
            state.total_incidents += 1;
            state.active_incidents += 1;
            state.last_incident_created = Some(SystemTime::now());
        }

        Ok(incident)
    }

    /// Assign an incident commander
    pub fn assign_commander(&self, incident_id: &str, commander_id: &str) -> SklResult<()> {
        let mut incidents = self.active_incidents.write()
            .map_err(|_| SklearsError::Other("Failed to acquire incidents lock".into()))?;

        if let Some(incident) = incidents.get_mut(incident_id) {
            incident.commander_assigned = Some(commander_id.to_string());
            incident.timeline.push(IncidentTimelineEntry {
                timestamp: SystemTime::now(),
                event_type: TimelineEventType::CommanderAssigned,
                description: format!("Commander {} assigned to incident", commander_id),
                actor: "incident_management_system".to_string(),
                details: {
                    let mut details = HashMap::new();
                    details.insert("commander_id".to_string(), commander_id.to_string());
                    details
                },
            });

            // Create command assignment
            let assignment = CommandAssignment {
                assignment_id: uuid::Uuid::new_v4().to_string(),
                incident_id: incident_id.to_string(),
                commander_id: commander_id.to_string(),
                assigned_at: SystemTime::now(),
                status: CommandStatus::Active,
                authority_level: AuthorityLevel::Full,
                responsibilities: vec![
                    "Incident coordination".to_string(),
                    "Response team management".to_string(),
                    "Stakeholder communication".to_string(),
                    "Resource allocation approval".to_string(),
                ],
            };

            let mut assignments = self.command_assignments.write()
                .map_err(|_| SklearsError::Other("Failed to acquire assignments lock".into()))?;
            assignments.insert(assignment.assignment_id.clone(), assignment);

            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!("Incident {} not found", incident_id)))
        }
    }

    /// Update incident status
    pub fn update_incident_status(&self, incident_id: &str, new_status: IncidentStatus) -> SklResult<()> {
        let mut incidents = self.active_incidents.write()
            .map_err(|_| SklearsError::Other("Failed to acquire incidents lock".into()))?;

        if let Some(incident) = incidents.get_mut(incident_id) {
            let old_status = incident.status.clone();
            incident.status = new_status.clone();

            incident.timeline.push(IncidentTimelineEntry {
                timestamp: SystemTime::now(),
                event_type: TimelineEventType::StatusChanged,
                description: format!("Status changed from {:?} to {:?}", old_status, new_status),
                actor: "incident_commander".to_string(),
                details: {
                    let mut details = HashMap::new();
                    details.insert("old_status".to_string(), format!("{:?}", old_status));
                    details.insert("new_status".to_string(), format!("{:?}", new_status));
                    details
                },
            });

            // If incident is resolved or closed, move to history and update coordination state
            if matches!(new_status, IncidentStatus::Resolved | IncidentStatus::Closed) {
                let mut state = self.coordination_state.write()
                    .map_err(|_| SklearsError::Other("Failed to acquire coordination state lock".into()))?;
                state.active_incidents = state.active_incidents.saturating_sub(1);
                state.resolved_incidents += 1;

                if incident.status == IncidentStatus::Resolved {
                    incident.actual_resolution_time = Some(
                        SystemTime::now().duration_since(incident.created_at)
                            .unwrap_or(Duration::from_secs(0))
                    );
                }

                // Archive to history
                let historical = HistoricalIncident {
                    incident: incident.clone(),
                    archived_at: SystemTime::now(),
                    final_status: new_status,
                    total_duration: SystemTime::now().duration_since(incident.created_at)
                        .unwrap_or(Duration::from_secs(0)),
                    post_incident_review_completed: false,
                };

                let mut history = self.incident_history.write()
                    .map_err(|_| SklearsError::Other("Failed to acquire history lock".into()))?;
                history.push(historical);
            }

            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!("Incident {} not found", incident_id)))
        }
    }

    /// Add a timeline entry to an incident
    pub fn add_timeline_entry(&self, incident_id: &str, entry: IncidentTimelineEntry) -> SklResult<()> {
        let mut incidents = self.active_incidents.write()
            .map_err(|_| SklearsError::Other("Failed to acquire incidents lock".into()))?;

        if let Some(incident) = incidents.get_mut(incident_id) {
            incident.timeline.push(entry);
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!("Incident {} not found", incident_id)))
        }
    }

    /// Record a resolution action
    pub fn record_resolution_action(&self, incident_id: &str, action: ResolutionAction) -> SklResult<()> {
        let mut incidents = self.active_incidents.write()
            .map_err(|_| SklearsError::Other("Failed to acquire incidents lock".into()))?;

        if let Some(incident) = incidents.get_mut(incident_id) {
            incident.resolution_actions.push(action.clone());
            incident.timeline.push(IncidentTimelineEntry {
                timestamp: SystemTime::now(),
                event_type: TimelineEventType::ResolutionAction,
                description: format!("Resolution action: {}", action.description),
                actor: action.executed_by.clone(),
                details: {
                    let mut details = HashMap::new();
                    details.insert("action_type".to_string(), format!("{:?}", action.action_type));
                    details.insert("action_id".to_string(), action.action_id.clone());
                    details
                },
            });
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!("Incident {} not found", incident_id)))
        }
    }

    /// Get active incidents
    pub fn get_active_incidents(&self) -> SklResult<Vec<Incident>> {
        let incidents = self.active_incidents.read()
            .map_err(|_| SklearsError::Other("Failed to acquire incidents lock".into()))?;
        Ok(incidents.values().cloned().collect())
    }

    /// Get incident by ID
    pub fn get_incident(&self, incident_id: &str) -> SklResult<Option<Incident>> {
        let incidents = self.active_incidents.read()
            .map_err(|_| SklearsError::Other("Failed to acquire incidents lock".into()))?;
        Ok(incidents.get(incident_id).cloned())
    }

    /// Get coordination state
    pub fn get_coordination_state(&self) -> SklResult<IncidentCoordinationState> {
        let state = self.coordination_state.read()
            .map_err(|_| SklearsError::Other("Failed to acquire coordination state lock".into()))?;
        Ok(state.clone())
    }

    /// Get command assignments
    pub fn get_command_assignments(&self) -> SklResult<Vec<CommandAssignment>> {
        let assignments = self.command_assignments.read()
            .map_err(|_| SklearsError::Other("Failed to acquire assignments lock".into()))?;
        Ok(assignments.values().cloned().collect())
    }

    /// Get incident history
    pub fn get_incident_history(&self, limit: Option<usize>) -> SklResult<Vec<HistoricalIncident>> {
        let history = self.incident_history.read()
            .map_err(|_| SklearsError::Other("Failed to acquire history lock".into()))?;

        if let Some(limit) = limit {
            Ok(history.iter().rev().take(limit).cloned().collect())
        } else {
            Ok(history.clone())
        }
    }

    fn setup_default_command_structure(&self) -> SklResult<()> {
        let mut state = self.coordination_state.write()
            .map_err(|_| SklearsError::Other("Failed to acquire coordination state lock".into()))?;

        state.available_commanders = vec![
            "incident_commander_primary".to_string(),
            "incident_commander_secondary".to_string(),
            "incident_commander_escalation".to_string(),
        ];

        Ok(())
    }
}

/// Incident structure with comprehensive tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Incident {
    pub incident_id: String,
    pub emergency_event_id: String,
    pub title: String,
    pub description: String,
    pub severity: EmergencySeverity,
    pub status: IncidentStatus,
    pub created_at: SystemTime,
    pub commander_assigned: Option<String>,
    pub response_team: Vec<String>,
    pub estimated_resolution: Option<Duration>,
    pub actual_resolution_time: Option<Duration>,
    pub root_cause: Option<String>,
    pub lessons_learned: Vec<String>,
    pub timeline: Vec<IncidentTimelineEntry>,
    pub impact_assessment: IncidentImpact,
    pub communication_log: Vec<CommunicationEntry>,
    pub escalation_history: Vec<String>,
    pub resolution_actions: Vec<ResolutionAction>,
}

/// Incident status tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IncidentStatus {
    Active,
    InProgress,
    Investigating,
    Mitigating,
    Monitoring,
    Resolved,
    Closed,
}

/// Incident timeline entry for comprehensive tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentTimelineEntry {
    pub timestamp: SystemTime,
    pub event_type: TimelineEventType,
    pub description: String,
    pub actor: String,
    pub details: HashMap<String, String>,
}

/// Timeline event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineEventType {
    IncidentCreated,
    CommanderAssigned,
    TeamDeployed,
    StatusChanged,
    EscalationTriggered,
    CommunicationSent,
    ResolutionAction,
    IncidentResolved,
    IncidentClosed,
}

/// Incident impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentImpact {
    pub affected_users: u64,
    pub affected_systems: Vec<String>,
    pub business_impact: BusinessImpactLevel,
    pub estimated_revenue_loss: Option<f64>,
    pub sla_violations: Vec<String>,
}

/// Business impact severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusinessImpactLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
    Catastrophic,
}

/// Communication log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationEntry {
    pub timestamp: SystemTime,
    pub communication_type: CommunicationType,
    pub sender: String,
    pub recipients: Vec<String>,
    pub subject: String,
    pub content: String,
    pub channels: Vec<String>,
}

/// Communication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationType {
    StatusUpdate,
    Escalation,
    Resolution,
    Investigation,
    Coordination,
}

/// Resolution action tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionAction {
    pub action_id: String,
    pub action_type: ResolutionActionType,
    pub description: String,
    pub executed_by: String,
    pub executed_at: SystemTime,
    pub duration: Option<Duration>,
    pub success: bool,
    pub impact: String,
    pub verification_required: bool,
}

/// Types of resolution actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionActionType {
    SystemRestart,
    ServiceRestart,
    ConfigurationChange,
    TrafficRedirection,
    ResourceScaling,
    DataRecovery,
    SecurityMitigation,
    Manual,
    Investigation,
}

/// Command assignment for incident commanders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandAssignment {
    pub assignment_id: String,
    pub incident_id: String,
    pub commander_id: String,
    pub assigned_at: SystemTime,
    pub status: CommandStatus,
    pub authority_level: AuthorityLevel,
    pub responsibilities: Vec<String>,
}

/// Command status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandStatus {
    Active,
    Transferred,
    Completed,
    Escalated,
}

/// Authority levels for incident commanders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorityLevel {
    Limited,
    Standard,
    Full,
    Executive,
}

/// Historical incident for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalIncident {
    pub incident: Incident,
    pub archived_at: SystemTime,
    pub final_status: IncidentStatus,
    pub total_duration: Duration,
    pub post_incident_review_completed: bool,
}

/// Incident coordination state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentCoordinationState {
    pub total_incidents: u64,
    pub active_incidents: u64,
    pub resolved_incidents: u64,
    pub average_resolution_time: Duration,
    pub last_incident_created: Option<SystemTime>,
    pub available_commanders: Vec<String>,
    pub coordination_metrics: CoordinationMetrics,
}

impl IncidentCoordinationState {
    pub fn new() -> Self {
        Self {
            total_incidents: 0,
            active_incidents: 0,
            resolved_incidents: 0,
            average_resolution_time: Duration::from_secs(0),
            last_incident_created: None,
            available_commanders: Vec::new(),
            coordination_metrics: CoordinationMetrics::default(),
        }
    }
}

/// Coordination effectiveness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMetrics {
    pub commander_response_time: Duration,
    pub team_mobilization_time: Duration,
    pub communication_efficiency: f64,
    pub resolution_effectiveness: f64,
    pub escalation_accuracy: f64,
}

impl Default for CoordinationMetrics {
    fn default() -> Self {
        Self {
            commander_response_time: Duration::from_minutes(5),
            team_mobilization_time: Duration::from_minutes(15),
            communication_efficiency: 0.85,
            resolution_effectiveness: 0.75,
            escalation_accuracy: 0.90,
        }
    }
}

impl Default for IncidentCommander {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incident_commander_creation() {
        let commander = IncidentCommander::new();
        assert!(commander.initialize().is_ok());
    }

    #[test]
    fn test_incident_creation() {
        let commander = IncidentCommander::new();
        commander.initialize().unwrap();

        let event = EmergencyEvent {
            event_id: "emergency-001".to_string(),
            emergency_type: EmergencyType::SystemFailure,
            severity: EmergencySeverity::Critical,
            title: "System Down".to_string(),
            description: "Primary system is unresponsive".to_string(),
            source: "health_monitor".to_string(),
            timestamp: SystemTime::now(),
            affected_systems: vec!["primary_system".to_string()],
            estimated_impact: super::super::detection::EmergencyImpact {
                user_impact: super::super::detection::UserImpact::High,
                business_impact: super::super::detection::BusinessImpact::High,
                system_impact: super::super::detection::SystemImpact::Critical,
                financial_impact: Some(10000.0),
            },
            estimated_impact_duration: Some(Duration::from_hours(2)),
            detected_by: "detector".to_string(),
            context: std::collections::HashMap::new(),
            related_events: vec![],
            urgency: super::super::detection::Urgency::High,
            requires_immediate_action: true,
        };

        let incident = commander.create_incident(&event).unwrap();
        assert_eq!(incident.emergency_event_id, "emergency-001");
        assert_eq!(incident.status, IncidentStatus::Active);
        assert!(!incident.timeline.is_empty());
    }

    #[test]
    fn test_incident_status_update() {
        let commander = IncidentCommander::new();
        commander.initialize().unwrap();

        let event = EmergencyEvent {
            event_id: "emergency-002".to_string(),
            emergency_type: EmergencyType::SystemFailure,
            severity: EmergencySeverity::High,
            title: "Service Degraded".to_string(),
            description: "Service experiencing high latency".to_string(),
            source: "health_monitor".to_string(),
            timestamp: SystemTime::now(),
            affected_systems: vec!["api_service".to_string()],
            estimated_impact: super::super::detection::EmergencyImpact {
                user_impact: super::super::detection::UserImpact::Medium,
                business_impact: super::super::detection::BusinessImpact::Medium,
                system_impact: super::super::detection::SystemImpact::High,
                financial_impact: Some(5000.0),
            },
            estimated_impact_duration: Some(Duration::from_hours(1)),
            detected_by: "detector".to_string(),
            context: std::collections::HashMap::new(),
            related_events: vec![],
            urgency: super::super::detection::Urgency::Medium,
            requires_immediate_action: false,
        };

        let incident = commander.create_incident(&event).unwrap();
        let result = commander.update_incident_status(&incident.incident_id, IncidentStatus::InProgress);
        assert!(result.is_ok());
    }

    #[test]
    fn test_commander_assignment() {
        let commander = IncidentCommander::new();
        commander.initialize().unwrap();

        let event = EmergencyEvent {
            event_id: "emergency-003".to_string(),
            emergency_type: EmergencyType::SecurityIncident,
            severity: EmergencySeverity::Critical,
            title: "Security Breach".to_string(),
            description: "Unauthorized access detected".to_string(),
            source: "security_monitor".to_string(),
            timestamp: SystemTime::now(),
            affected_systems: vec!["user_database".to_string()],
            estimated_impact: super::super::detection::EmergencyImpact {
                user_impact: super::super::detection::UserImpact::Critical,
                business_impact: super::super::detection::BusinessImpact::Critical,
                system_impact: super::super::detection::SystemImpact::High,
                financial_impact: Some(50000.0),
            },
            estimated_impact_duration: Some(Duration::from_hours(4)),
            detected_by: "security_detector".to_string(),
            context: std::collections::HashMap::new(),
            related_events: vec![],
            urgency: super::super::detection::Urgency::Critical,
            requires_immediate_action: true,
        };

        let incident = commander.create_incident(&event).unwrap();
        let result = commander.assign_commander(&incident.incident_id, "security_commander");
        assert!(result.is_ok());

        let assignments = commander.get_command_assignments().unwrap();
        assert!(!assignments.is_empty());
    }
}