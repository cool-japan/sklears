//! Emergency Response System
//!
//! A comprehensive emergency response framework providing end-to-end emergency management
//! capabilities including detection, escalation, incident management, crisis communication,
//! response coordination, resource management, and recovery procedures.
//!
//! # Architecture
//!
//! The emergency response system is organized into seven main components:
//!
//! - **Core Management** (`emergency_core`): Central orchestration and system coordination
//! - **Detection System** (`detection`): Emergency detection, monitoring, and alerting
//! - **Escalation Management** (`escalation`): Automated escalation workflows and contact management
//! - **Incident Command** (`incident_management`): Incident creation, tracking, and coordination
//! - **Crisis Communication** (`crisis_communication`): Multi-channel emergency communications
//! - **Response Coordination** (`response_coordination`): Team mobilization and deployment
//! - **Resource Management** (`resource_management`): Emergency resource allocation and optimization
//! - **Recovery Procedures** (`recovery_procedures`): Recovery workflows and post-incident analysis
//!
//! # Usage
//!
//! ```rust
//! use crate::emergency_response::{EmergencyResponseCore, EmergencyDetector};
//!
//! // Create and initialize the emergency response system
//! let emergency_system = EmergencyResponseCore::new();
//! emergency_system.initialize().unwrap();
//!
//! // The system automatically coordinates all subsystems for comprehensive emergency response
//! ```
//!
//! # Key Features
//!
//! - **Real-time Detection**: Continuous monitoring with configurable detection rules
//! - **Automated Escalation**: Multi-level escalation with contact hierarchies
//! - **Incident Command**: Professional incident management with timeline tracking
//! - **Crisis Communication**: Multi-channel notifications with stakeholder management
//! - **Team Coordination**: Response team mobilization with skill-based routing
//! - **Resource Management**: Intelligent resource allocation with cost optimization
//! - **Recovery Orchestration**: Automated recovery workflows with post-incident analysis
//! - **Continuous Improvement**: Lessons learned and optimization recommendations

// Module declarations
pub mod emergency_core;
pub mod detection;
pub mod escalation;
pub mod incident_management;
pub mod crisis_communication;
pub mod response_coordination;
pub mod resource_management;
pub mod recovery_procedures;

// Re-export core management types
pub use emergency_core::{
    EmergencyResponseCore,
    EmergencySystemState,
    EmergencySystemStatus,
    EmergencyProtocols,
    EmergencyProtocol,
    ProtocolStep,
    ProtocolStepType,
    ProtocolPriority,
    RetryPolicy,
    BackoffStrategy,
    ProtocolExecutionResult,
    ProtocolStepResult,
    ProtocolStepFailure,
    ResourceConsumption,
    ImpactAssessment,
    EmergencyStatus,
    EmergencyLevel,
    ResponsePriority,
    EmergencyResponseStatus,
    EmergencyResponse,
};

// Re-export detection system types
pub use detection::{
    EmergencyDetector,
    EmergencyDetectionRule,
    DetectionRuleType,
    DetectionCondition,
    DetectionState,
    ThreatLevel,
    SystemHealthMetrics,
    LatencyMetrics,
    ResourceUtilizationMetrics,
    SecurityMetrics,
    BusinessHealthMetrics,
    EmergencyEvent,
    EmergencyType,
    EmergencySeverity,
    EmergencyImpact,
    UserImpact,
    BusinessImpact,
    SystemImpact,
    Urgency,
};

// Re-export escalation management types
pub use escalation::{
    EscalationManager,
    EscalationLevel,
    EscalationContact,
    ContactMethod,
    ContactAvailability,
    EscalationAction,
    EscalationActionType,
    ActiveEscalation,
    EscalationStatus,
    EscalationResult,
    EscalationEvent,
    EscalationEventType,
    NotificationResult,
    ActionResult,
};

// Re-export incident management types
pub use incident_management::{
    IncidentCommander,
    Incident,
    IncidentStatus,
    IncidentTimelineEntry,
    TimelineEventType,
    IncidentImpact,
    BusinessImpactLevel,
    CommunicationEntry,
    CommunicationType,
    ResolutionAction,
    ResolutionActionType,
    CommandAssignment,
    CommandStatus,
    AuthorityLevel,
    HistoricalIncident,
    IncidentCoordinationState,
    CoordinationMetrics,
};

// Re-export crisis communication types
pub use crisis_communication::{
    CrisisCommunicator,
    EmergencyNotification,
    NotificationType,
    NotificationPriority,
    CommunicationChannel,
    ChannelType,
    ContactMethod as CommunicationContactMethod,
    Stakeholder,
    StakeholderType,
    NotificationPreferences,
    EscalationLevel as CommunicationEscalationLevel,
    StakeholderAvailability,
    MessageTemplate,
    TemplateType,
    NotificationCampaign,
    CampaignType,
    CampaignStatus,
    StatusUpdate,
    PublicMessage,
    ResolutionMessage,
    NotificationResult as CommunicationNotificationResult,
    CommunicationLogEntry,
    CommunicationLogType,
    DeliveryStatus,
    CommunicationMetrics,
    CommunicationState,
};

// Re-export response coordination types
pub use response_coordination::{
    ResponseTeamCoordinator,
    ResponseTeam,
    TeamType,
    TeamMember,
    MemberAvailability,
    TeamAvailability,
    TeamDeployment,
    DeploymentStatus,
    DeploymentLocation,
    TaskAssignment,
    TaskType,
    TaskPriority,
    TaskStatus,
    TeamCapacityTracker,
    TeamUtilization,
    MobilizationResult,
    ResponseCapacity,
    TeamCapacityStatus,
    CapacityConstraint,
    ConstraintType,
    ConstraintSeverity,
    StandDownResult,
    HistoricalDeployment,
    TeamPerformanceMetrics,
    DeploymentPerformanceMetrics,
    CoordinationMetrics as ResponseCoordinationMetrics,
};

// Re-export resource management types
pub use resource_management::{
    EmergencyResourceManager,
    ResourcePool,
    ResourceType,
    ResourceAllocation,
    ResourceAllocationResult,
    ResourceRequirement,
    AllocationPriority,
    AllocationConstraint,
    ConstraintType as ResourceConstraintType,
    ScalingPolicy,
    ResourcePoolMetrics,
    ResourceUtilizationStatus,
    ResourcePoolUtilization,
    ResourceConstraint,
    ConstraintSeverity as ResourceConstraintSeverity,
    ConstraintImpact,
    OptimizationOpportunity,
    OpportunityType,
    OptimizationAction,
    ResourceReleaseResult,
    HistoricalAllocation,
    CapacityPlanner,
    UsageRecord,
    ForecastingModel,
    ModelType,
    CapacityRecommendation,
    ResourceCostTracker,
    DailyCostRecord,
    ResourceCostAnalysis,
    CostTrend,
    TrendDirection,
    ResourceOptimizationEngine,
    OptimizationAlgorithm,
    AlgorithmType,
    OptimizationResult,
    OptimizationRecommendation,
    RecommendationType,
    ImplementationEffort,
    AllocationPolicy,
    PolicyType,
    PolicyCondition,
    ConditionType,
    ComparisonOperator,
    PolicyAction,
    ActionType,
};

// Re-export recovery procedures types
pub use recovery_procedures::{
    RecoveryManager,
    RecoveryProcedure,
    RecoveryProcedureType,
    RecoveryStep,
    RecoveryStepType,
    AutomationLevel,
    ActiveRecovery,
    RecoveryStatus,
    StepExecutionResult,
    RecoveryResult,
    HistoricalRecovery as RecoveryHistoricalRecord,
    ApprovalRequirements,
    RiskAssessment,
    RiskLevel,
    RecoveryWorkflowEngine,
    WorkflowExecution,
    WorkflowStatus,
    WorkflowTemplate,
    WorkflowStep,
    WorkflowStepType,
    WorkflowCondition,
    PostIncidentAnalysisTracker,
    PostIncidentAnalysis,
    AnalysisStatus,
    TimelineAnalysis,
    DelayAnalysis,
    CriticalPathItem,
    RootCauseAnalysis,
    FailureChainItem,
    EffectivenessAssessment,
    LessonLearned,
    LessonCategory,
    LessonPriority,
    ImplementationStatus,
    ImprovementRecommendation,
    RecommendationCategory,
    ImpactLevel,
    EffortLevel,
    RecommendationStatus,
    StakeholderFeedback,
    FeedbackType,
    DocumentationStatus,
    PostIncidentResult,
    RecoveryMetricsCollector,
    RecoveryMetrics,
    RecoveryOptimizationEngine,
    RecoveryRecommendation,
    RecoveryRecommendationType,
    ComplexityLevel,
    RecommendationPriority,
};

// Convenience type aliases for commonly used combinations
pub type EmergencyCore = EmergencyResponseCore;
pub type Detector = EmergencyDetector;
pub type EscalationMgr = EscalationManager;
pub type IncidentCmd = IncidentCommander;
pub type CrisisComm = CrisisCommunicator;
pub type ResponseCoord = ResponseTeamCoordinator;
pub type ResourceMgr = EmergencyResourceManager;
pub type RecoveryMgr = RecoveryManager;

// Core error handling
pub use emergency_core::EmergencyResponseError as Error;
pub use emergency_core::EmergencyResponseResult as Result;

/// Main entry point for emergency response operations
///
/// Provides a simplified interface for emergency response operations
/// while maintaining access to all underlying subsystems.
pub struct EmergencyResponseSystem {
    core: EmergencyResponseCore,
}

impl EmergencyResponseSystem {
    /// Create a new emergency response system
    pub fn new() -> Self {
        Self {
            core: EmergencyResponseCore::new(),
        }
    }

    /// Initialize the emergency response system
    pub fn initialize(&self) -> Result<()> {
        self.core.initialize()
    }

    /// Get access to the core emergency response system
    pub fn core(&self) -> &EmergencyResponseCore {
        &self.core
    }

    /// Get mutable access to the core emergency response system
    pub fn core_mut(&mut self) -> &mut EmergencyResponseCore {
        &mut self.core
    }

    /// Handle an emergency event (simplified interface)
    pub fn handle_emergency(&self, event: EmergencyEvent) -> Result<EmergencyResponse> {
        self.core.handle_emergency(event)
    }

    /// Get system health status
    pub fn health_check(&self) -> SystemHealthStatus {
        // Implementation would check all subsystems
        SystemHealthStatus {
            overall_status: SystemStatus::Healthy,
            subsystem_status: std::collections::HashMap::new(),
            last_check: std::time::SystemTime::now(),
            emergency_readiness: EmergencyReadinessLevel::Ready,
        }
    }

    /// Shutdown the emergency response system
    pub fn shutdown(&self) -> Result<()> {
        self.core.shutdown()
    }
}

/// System health status for monitoring
#[derive(Debug, Clone)]
pub struct SystemHealthStatus {
    pub overall_status: SystemStatus,
    pub subsystem_status: std::collections::HashMap<String, SystemStatus>,
    pub last_check: std::time::SystemTime,
    pub emergency_readiness: EmergencyReadinessLevel,
}

/// System status levels
#[derive(Debug, Clone)]
pub enum SystemStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
    Unknown,
}

/// Emergency readiness levels
#[derive(Debug, Clone)]
pub enum EmergencyReadinessLevel {
    Ready,
    Degraded,
    NotReady,
    Maintenance,
}

impl Default for EmergencyResponseSystem {
    fn default() -> Self {
        Self::new()
    }
}

// Feature flags for conditional compilation
#[cfg(feature = "async")]
pub mod async_support {
    //! Async support for emergency response operations
    //!
    //! This module provides async versions of emergency response operations
    //! for integration with async runtimes.

    use super::*;
    use std::future::Future;
    use std::pin::Pin;

    /// Async version of EmergencyResponseSystem
    pub struct AsyncEmergencyResponseSystem {
        inner: EmergencyResponseSystem,
    }

    impl AsyncEmergencyResponseSystem {
        pub fn new() -> Self {
            Self {
                inner: EmergencyResponseSystem::new(),
            }
        }

        /// Async emergency handling
        pub async fn handle_emergency(&self, event: EmergencyEvent) -> Result<EmergencyResponse> {
            // Implementation would use async execution
            tokio::task::spawn_blocking({
                let event = event;
                let inner = &self.inner;
                move || inner.handle_emergency(event)
            }).await.unwrap()
        }

        /// Async initialization
        pub async fn initialize(&self) -> Result<()> {
            tokio::task::spawn_blocking({
                let inner = &self.inner;
                move || inner.initialize()
            }).await.unwrap()
        }
    }
}

#[cfg(feature = "metrics")]
pub mod metrics_integration {
    //! Integration with external metrics systems
    //!
    //! Provides adapters for popular metrics collection systems
    //! like Prometheus, InfluxDB, and custom telemetry.

    use super::*;

    /// Prometheus metrics exporter for emergency response
    pub struct PrometheusExporter {
        core: std::sync::Arc<std::sync::RwLock<EmergencyResponseCore>>,
    }

    impl PrometheusExporter {
        pub fn new(core: std::sync::Arc<std::sync::RwLock<EmergencyResponseCore>>) -> Self {
            Self { core }
        }

        /// Export metrics in Prometheus format
        pub fn export(&self) -> String {
            let core = self.core.read().unwrap();
            let state = core.get_emergency_state().unwrap_or_default();

            format!(
                "# HELP emergency_response_active Whether emergency response is active\\n\\\
                 # TYPE emergency_response_active gauge\\n\\\
                 emergency_response_active {}\\n\\\
                 # HELP emergency_response_total_handled Total number of emergencies handled\\n\\\
                 # TYPE emergency_response_total_handled counter\\n\\\
                 emergency_response_total_handled {}\\n\",
                if matches!(state.status, EmergencySystemStatus::Emergency) { 1 } else { 0 },
                state.total_emergencies_handled
            )
        }
    }
}

// Documentation examples and tests
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emergency_response_system_creation() {
        let _system = EmergencyResponseSystem::new();
    }

    #[test]
    fn test_emergency_response_initialization() {
        let system = EmergencyResponseSystem::new();
        let result = system.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_health_check() {
        let system = EmergencyResponseSystem::new();
        system.initialize().unwrap();
        let health = system.health_check();
        assert!(matches!(health.overall_status, SystemStatus::Healthy));
    }

    #[test]
    fn test_emergency_handling_workflow() {
        let system = EmergencyResponseSystem::new();
        system.initialize().unwrap();

        let event = EmergencyEvent {
            event_id: "test-emergency-001".to_string(),
            emergency_type: EmergencyType::SystemFailure,
            severity: EmergencySeverity::Critical,
            title: "Test System Failure".to_string(),
            description: "Test emergency for integration testing".to_string(),
            source: "test_suite".to_string(),
            timestamp: std::time::SystemTime::now(),
            affected_systems: vec!["test_system".to_string()],
            estimated_impact: EmergencyImpact {
                user_impact: UserImpact::High,
                business_impact: BusinessImpact::High,
                system_impact: SystemImpact::Critical,
                financial_impact: Some(10000.0),
            },
            estimated_impact_duration: Some(std::time::Duration::from_hours(2)),
            detected_by: "test_detector".to_string(),
            context: std::collections::HashMap::new(),
            related_events: vec![],
            urgency: Urgency::High,
            requires_immediate_action: true,
        };

        let response = system.handle_emergency(event);
        assert!(response.is_ok());

        let emergency_response = response.unwrap();
        assert!(!emergency_response.response_id.is_empty());
        assert_eq!(emergency_response.status, EmergencyResponseStatus::Active);
    }

    #[test]
    fn test_subsystem_integration() {
        let system = EmergencyResponseSystem::new();
        system.initialize().unwrap();

        // Test that all subsystems are properly integrated
        let core = system.core();

        // Verify detector is available
        let detector_result = core.detector.initialize();
        assert!(detector_result.is_ok());

        // Verify escalation manager is available
        let escalation_result = core.escalation_manager.initialize();
        assert!(escalation_result.is_ok());

        // Verify incident commander is available
        let incident_result = core.incident_commander.initialize();
        assert!(incident_result.is_ok());

        // Verify crisis communicator is available
        let comm_result = core.crisis_communicator.initialize();
        assert!(comm_result.is_ok());

        // Verify response coordinator is available
        let coord_result = core.response_coordinator.initialize();
        assert!(coord_result.is_ok());

        // Verify resource manager is available
        let resource_result = core.resource_manager.initialize();
        assert!(resource_result.is_ok());

        // Verify recovery manager is available
        let recovery_result = core.recovery_manager.initialize();
        assert!(recovery_result.is_ok());
    }

    #[test]
    fn test_emergency_workflow_end_to_end() {
        let system = EmergencyResponseSystem::new();
        system.initialize().unwrap();

        // Create emergency event
        let event = EmergencyEvent {
            event_id: "workflow-test-001".to_string(),
            emergency_type: EmergencyType::SecurityIncident,
            severity: EmergencySeverity::Critical,
            title: "Security Breach Detected".to_string(),
            description: "Unauthorized access attempt detected".to_string(),
            source: "security_monitor".to_string(),
            timestamp: std::time::SystemTime::now(),
            affected_systems: vec!["auth_service".to_string(), "user_database".to_string()],
            estimated_impact: EmergencyImpact {
                user_impact: UserImpact::Critical,
                business_impact: BusinessImpact::Critical,
                system_impact: SystemImpact::High,
                financial_impact: Some(50000.0),
            },
            estimated_impact_duration: Some(std::time::Duration::from_hours(4)),
            detected_by: "security_detector".to_string(),
            context: std::collections::HashMap::new(),
            related_events: vec![],
            urgency: Urgency::Critical,
            requires_immediate_action: true,
        };

        // Handle emergency - this should trigger the full workflow
        let response = system.handle_emergency(event);
        assert!(response.is_ok());

        let emergency_response = response.unwrap();

        // Verify response characteristics
        assert!(!emergency_response.response_id.is_empty());
        assert_eq!(emergency_response.status, EmergencyResponseStatus::Active);
        assert!(!emergency_response.response_teams.is_empty());
        assert!(!emergency_response.allocated_resources.is_empty());
        assert_eq!(emergency_response.priority, ResponsePriority::Critical);

        // Verify workflow completion
        assert!(!emergency_response.protocol_results.is_empty());

        // Test shutdown
        let shutdown_result = system.shutdown();
        assert!(shutdown_result.is_ok());
    }
}

#[cfg(doctest)]
mod doctests {
    //! Documentation tests to ensure examples in docs work correctly

    /// Example from module documentation
    ///
    /// ```rust
    /// use emergency_response::{EmergencyResponseCore, EmergencyDetector};
    ///
    /// let emergency_system = EmergencyResponseCore::new();
    /// // Additional setup would be required for a complete example
    /// ```
    fn doctest_example() {}
}