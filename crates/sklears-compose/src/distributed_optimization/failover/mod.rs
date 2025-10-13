//! Comprehensive failover system modules
//!
//! This module contains the complete distributed failover and disaster recovery system,
//! systematically organized into focused, maintainable modules following the 2000-line policy.
//!
//! # Architecture Overview
//!
//! The failover system is organized into 8 specialized modules:
//!
//! - **`failover_strategies`**: Comprehensive strategy management with triggers and validation
//! - **`failover_coordination`**: Central coordination with policies and notifications
//! - **`execution_orchestration`**: Advanced orchestration engine with optimization
//! - **`workflow_management`**: Workflow definitions, templates, and analytics
//! - **`dependency_resolution`**: Graph analysis and circular dependency handling
//! - **`split_brain_detection`**: Split-brain detection with quorum and fencing systems
//! - **`state_coordination`**: State machine management and coordination protocols
//! - **`compliance_reporting`**: Compliance checking, reporting, and dashboards
//!
//! # Module Breakdown
//!
//! ## Strategy Management (~400+ lines)
//! Comprehensive failover strategy definitions, trigger conditions, execution steps,
//! rollback procedures, and validation mechanisms.
//!
//! ## Coordination System (~450+ lines)
//! Central coordination with policies, active failover tracking, notification engines,
//! and escalation matrices for enterprise environments.
//!
//! ## Orchestration Engine (~500+ lines)
//! Advanced execution orchestration with plan optimization, resource scheduling,
//! conflict resolution, and performance analysis.
//!
//! ## Workflow Management (~450+ lines)
//! Comprehensive workflow definitions, active workflow tracking, templates,
//! and detailed analytics with bottleneck analysis.
//!
//! ## Dependency Resolution (~400+ lines)
//! Sophisticated dependency graph analysis, circular dependency detection,
//! resolution algorithms, and compliance validation.
//!
//! ## Split-Brain Detection (~400+ lines)
//! Advanced split-brain detection algorithms, quorum systems, fencing mechanisms,
//! and comprehensive safety systems.
//!
//! ## State Coordination (~350+ lines)
//! Failover state machine management, coordination protocols, consensus tracking,
//! and distributed state synchronization.
//!
//! ## Compliance Reporting (~350+ lines)
//! Compliance framework management, automated reporting, interactive dashboards,
//! and comprehensive analytics with benchmarking.

/// Comprehensive failover strategy management system
pub mod failover_strategies;

/// Central failover coordination and notification system
pub mod failover_coordination;

/// Advanced execution orchestration and optimization engine
pub mod execution_orchestration;

/// Workflow management with templates and analytics
pub mod workflow_management;

/// Dependency resolution with graph analysis and circular dependency handling
pub mod dependency_resolution;

/// Split-brain detection with quorum systems and fencing mechanisms
pub mod split_brain_detection;

/// State machine management and coordination protocols
pub mod state_coordination;

/// Compliance checking, reporting, and dashboards
pub mod compliance_reporting;

// Re-export key structures for convenient access
pub use failover_strategies::{
    FailoverStrategy, FailoverTrigger, FailoverStep, RollbackStep, ValidationCheck,
    FailoverStrategyType, TriggerType, TriggerCondition, StepType, StepAction,
    ValidationCriteria, StrategyConfig, StepTimeout, ValidationTimeout,
};

pub use failover_coordination::{
    FailoverCoordinator, FailoverPolicies, NotificationEngine, EscalationMatrix,
    ActiveFailover, FailoverHistory, CoordinationState, NotificationPolicies,
    EscalationLevel, EscalationAction, NotificationChannel, NotificationTarget,
};

pub use execution_orchestration::{
    OrchestrationEngine, ExecutionPlan, PlanOptimizer, ResourceScheduler, ConflictResolver,
    ExecutionGraph, ExecutionNode, ExecutionEdge, ResourceRequirements, TimingConstraints,
    RiskAssessment, OptimizationObjective, OptimizationConstraint, OptimizationAlgorithm,
};

pub use workflow_management::{
    WorkflowManager, WorkflowDefinition, ActiveWorkflow, WorkflowTemplate, WorkflowAnalytics,
    WorkflowStep, WorkflowTrigger, WorkflowOutput, WorkflowState, ExecutionContext,
    WorkflowStepType, WorkflowTriggerType, OutputType, OutputFormat, TemplateCategory,
};

pub use dependency_resolution::{
    DependencyResolver, DependencyGraph, DependencyNode, DependencyEdge, CircularDependencyHandler,
    DependencyValidation, DependencyNodeType, DependencyType, ResolutionAlgorithm,
    CircularDetectionAlgorithm, CircularBreakingStrategy, CircularPreventionMeasure,
};

pub use split_brain_detection::{
    SplitBrainDetector, QuorumSystem, FencingMechanism, DetectionConfig, ResolutionStrategy,
    SplitBrainAlgorithm, QuorumType, FencingType, FencingConfig, ResolutionType, ResolutionStep,
    TieBreakingRule, TieBreakingType, QuorumManager, FencingCoordinator,
};

pub use state_coordination::{
    FailoverStateMachine, CoordinationProtocol, StateDefinition, StateTransition, StateValidator,
    ProtocolConfig, SecurityConfig, MessageHandler, CoordinationState, ParticipantStatus,
    StateType, ProtocolType, SecurityLevel, StateManager, CoordinationManager,
};

pub use compliance_reporting::{
    ComplianceChecking, ComplianceFramework, ComplianceAssessment, ComplianceReporting,
    ComplianceRequirement, ComplianceFinding, ReportTemplate, AutomatedReporting,
    ComplianceDashboard, DashboardWidget, RequirementCategory, ComplianceLevel,
    VerificationMethod, FindingType, FindingSeverity, ReportFormat, SectionType,
};