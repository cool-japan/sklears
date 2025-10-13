use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringPolicies {
    pub policy_framework: PolicyFramework,
    pub configuration_management: ConfigurationManagement,
    pub compliance_enforcement: ComplianceEnforcement,
    pub governance_structure: GovernanceStructure,
    pub policy_analytics: PolicyAnalytics,
    pub audit_framework: AuditFramework,
    pub lifecycle_management: LifecycleManagement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyFramework {
    pub policy_definitions: HashMap<String, PolicyDefinition>,
    pub policy_hierarchy: PolicyHierarchy,
    pub policy_templates: HashMap<String, PolicyTemplate>,
    pub policy_enforcement: PolicyEnforcement,
    pub policy_validation: PolicyValidation,
    pub policy_versioning: PolicyVersioning,
    pub policy_inheritance: PolicyInheritance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDefinition {
    pub policy_id: String,
    pub policy_name: String,
    pub policy_description: String,
    pub policy_scope: PolicyScope,
    pub policy_rules: Vec<PolicyRule>,
    pub enforcement_level: EnforcementLevel,
    pub policy_metadata: PolicyMetadata,
    pub effective_period: EffectivePeriod,
    pub policy_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyScope {
    Global {
        applies_to_all: bool,
        global_overrides: Vec<String>,
    },
    Regional {
        regions: Vec<String>,
        regional_variations: HashMap<String, String>,
    },
    Organizational {
        organizational_units: Vec<String>,
        hierarchy_enforcement: bool,
    },
    Functional {
        functional_areas: Vec<String>,
        cross_functional_rules: Vec<String>,
    },
    Temporal {
        time_based_scope: TimeBasedScope,
        schedule_variations: Vec<ScheduleVariation>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationManagement {
    pub configuration_schema: ConfigurationSchema,
    pub configuration_validation: ConfigurationValidation,
    pub configuration_deployment: ConfigurationDeployment,
    pub configuration_monitoring: ConfigurationMonitoring,
    pub configuration_backup: ConfigurationBackup,
    pub configuration_synchronization: ConfigurationSynchronization,
    pub dynamic_configuration: DynamicConfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationSchema {
    pub schema_definitions: HashMap<String, SchemaDefinition>,
    pub data_types: HashMap<String, DataType>,
    pub validation_rules: Vec<ValidationRule>,
    pub schema_relationships: SchemaRelationships,
    pub schema_versioning: SchemaVersioning,
    pub schema_evolution: SchemaEvolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    String {
        max_length: Option<usize>,
        pattern_validation: Option<String>,
        allowed_values: Option<Vec<String>>,
    },
    Integer {
        min_value: Option<i64>,
        max_value: Option<i64>,
        step_size: Option<i64>,
    },
    Float {
        min_value: Option<f64>,
        max_value: Option<f64>,
        precision: Option<u32>,
    },
    Boolean {
        default_value: Option<bool>,
    },
    Array {
        element_type: Box<DataType>,
        min_items: Option<usize>,
        max_items: Option<usize>,
    },
    Object {
        properties: HashMap<String, DataType>,
        required_properties: Vec<String>,
        additional_properties: bool,
    },
    Enum {
        allowed_values: Vec<String>,
        case_sensitive: bool,
    },
    Duration {
        min_duration: Option<Duration>,
        max_duration: Option<Duration>,
        time_units: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceEnforcement {
    pub compliance_frameworks: HashMap<String, ComplianceFramework>,
    pub audit_requirements: AuditRequirements,
    pub compliance_monitoring: ComplianceMonitoring,
    pub violation_handling: ViolationHandling,
    pub compliance_reporting: ComplianceReporting,
    pub remediation_procedures: RemediationProcedures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFramework {
    pub framework_id: String,
    pub framework_name: String,
    pub compliance_standards: Vec<ComplianceStandard>,
    pub certification_requirements: CertificationRequirements,
    pub assessment_procedures: AssessmentProcedures,
    pub framework_mapping: FrameworkMapping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceStructure {
    pub governance_hierarchy: GovernanceHierarchy,
    pub decision_making: DecisionMaking,
    pub accountability_matrix: AccountabilityMatrix,
    pub approval_workflows: ApprovalWorkflows,
    pub governance_roles: GovernanceRoles,
    pub escalation_procedures: EscalationProcedures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GovernanceHierarchy {
    Centralized {
        central_authority: String,
        delegation_rules: Vec<DelegationRule>,
        override_mechanisms: Vec<OverrideMechanism>,
    },
    Federated {
        federation_members: Vec<FederationMember>,
        coordination_mechanisms: CoordinationMechanisms,
        conflict_resolution: ConflictResolution,
    },
    Distributed {
        autonomous_units: Vec<AutonomousUnit>,
        coordination_protocols: CoordinationProtocols,
        consensus_mechanisms: ConsensusMechanisms,
    },
    Hybrid {
        hierarchical_components: Vec<HierarchicalComponent>,
        integration_strategies: IntegrationStrategies,
        governance_interfaces: GovernanceInterfaces,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyAnalytics {
    pub analytics_framework: AnalyticsFramework,
    pub policy_metrics: PolicyMetrics,
    pub effectiveness_analysis: EffectivenessAnalysis,
    pub compliance_analytics: ComplianceAnalytics,
    pub trend_analysis: TrendAnalysis,
    pub predictive_analytics: PredictiveAnalytics,
    pub reporting_dashboard: ReportingDashboard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditFramework {
    pub audit_policies: HashMap<String, AuditPolicy>,
    pub audit_trails: AuditTrails,
    pub audit_procedures: AuditProcedures,
    pub audit_reporting: AuditReporting,
    pub audit_automation: AuditAutomation,
    pub external_audit_integration: ExternalAuditIntegration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleManagement {
    pub lifecycle_stages: Vec<LifecycleStage>,
    pub transition_management: TransitionManagement,
    pub deprecation_policies: DeprecationPolicies,
    pub migration_strategies: MigrationStrategies,
    pub lifecycle_governance: LifecycleGovernance,
    pub end_of_life_procedures: EndOfLifeProcedures,
}

impl Default for MonitoringPolicies {
    fn default() -> Self {
        Self {
            policy_framework: PolicyFramework::default(),
            configuration_management: ConfigurationManagement::default(),
            compliance_enforcement: ComplianceEnforcement::default(),
            governance_structure: GovernanceStructure::default(),
            policy_analytics: PolicyAnalytics::default(),
            audit_framework: AuditFramework::default(),
            lifecycle_management: LifecycleManagement::default(),
        }
    }
}

impl Default for PolicyFramework {
    fn default() -> Self {
        Self {
            policy_definitions: HashMap::new(),
            policy_hierarchy: PolicyHierarchy::default(),
            policy_templates: HashMap::new(),
            policy_enforcement: PolicyEnforcement::default(),
            policy_validation: PolicyValidation::default(),
            policy_versioning: PolicyVersioning::default(),
            policy_inheritance: PolicyInheritance::default(),
        }
    }
}

impl Default for ConfigurationManagement {
    fn default() -> Self {
        Self {
            configuration_schema: ConfigurationSchema::default(),
            configuration_validation: ConfigurationValidation::default(),
            configuration_deployment: ConfigurationDeployment::default(),
            configuration_monitoring: ConfigurationMonitoring::default(),
            configuration_backup: ConfigurationBackup::default(),
            configuration_synchronization: ConfigurationSynchronization::default(),
            dynamic_configuration: DynamicConfiguration::default(),
        }
    }
}

impl Default for ConfigurationSchema {
    fn default() -> Self {
        Self {
            schema_definitions: HashMap::new(),
            data_types: HashMap::new(),
            validation_rules: Vec::new(),
            schema_relationships: SchemaRelationships::default(),
            schema_versioning: SchemaVersioning::default(),
            schema_evolution: SchemaEvolution::default(),
        }
    }
}

impl Default for ComplianceEnforcement {
    fn default() -> Self {
        Self {
            compliance_frameworks: HashMap::new(),
            audit_requirements: AuditRequirements::default(),
            compliance_monitoring: ComplianceMonitoring::default(),
            violation_handling: ViolationHandling::default(),
            compliance_reporting: ComplianceReporting::default(),
            remediation_procedures: RemediationProcedures::default(),
        }
    }
}

impl Default for GovernanceStructure {
    fn default() -> Self {
        Self {
            governance_hierarchy: GovernanceHierarchy::Centralized {
                central_authority: "central_governance".to_string(),
                delegation_rules: Vec::new(),
                override_mechanisms: Vec::new(),
            },
            decision_making: DecisionMaking::default(),
            accountability_matrix: AccountabilityMatrix::default(),
            approval_workflows: ApprovalWorkflows::default(),
            governance_roles: GovernanceRoles::default(),
            escalation_procedures: EscalationProcedures::default(),
        }
    }
}

impl Default for PolicyAnalytics {
    fn default() -> Self {
        Self {
            analytics_framework: AnalyticsFramework::default(),
            policy_metrics: PolicyMetrics::default(),
            effectiveness_analysis: EffectivenessAnalysis::default(),
            compliance_analytics: ComplianceAnalytics::default(),
            trend_analysis: TrendAnalysis::default(),
            predictive_analytics: PredictiveAnalytics::default(),
            reporting_dashboard: ReportingDashboard::default(),
        }
    }
}

impl Default for AuditFramework {
    fn default() -> Self {
        Self {
            audit_policies: HashMap::new(),
            audit_trails: AuditTrails::default(),
            audit_procedures: AuditProcedures::default(),
            audit_reporting: AuditReporting::default(),
            audit_automation: AuditAutomation::default(),
            external_audit_integration: ExternalAuditIntegration::default(),
        }
    }
}

impl Default for LifecycleManagement {
    fn default() -> Self {
        Self {
            lifecycle_stages: Vec::new(),
            transition_management: TransitionManagement::default(),
            deprecation_policies: DeprecationPolicies::default(),
            migration_strategies: MigrationStrategies::default(),
            lifecycle_governance: LifecycleGovernance::default(),
            end_of_life_procedures: EndOfLifeProcedures::default(),
        }
    }
}

// Supporting types and enums
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    Mandatory,
    Recommended,
    Optional,
    Conditional { conditions: Vec<String> },
}

// Supporting structures with Default implementations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyRule {
    pub rule_id: String,
    pub rule_description: String,
    pub rule_logic: String,
    pub rule_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyMetadata {
    pub created_by: String,
    pub creation_date: Instant,
    pub last_modified: Instant,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EffectivePeriod {
    pub start_date: Option<Instant>,
    pub end_date: Option<Instant>,
    pub timezone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeBasedScope {
    pub time_windows: Vec<TimeWindow>,
    pub recurring_patterns: Vec<RecurringPattern>,
    pub exception_periods: Vec<ExceptionPeriod>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScheduleVariation {
    pub variation_id: String,
    pub applicable_schedule: String,
    pub variation_rules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyHierarchy {
    pub hierarchy_levels: Vec<String>,
    pub inheritance_rules: Vec<String>,
    pub conflict_resolution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyTemplate {
    pub template_id: String,
    pub template_structure: String,
    pub variable_definitions: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyEnforcement {
    pub enforcement_mechanisms: Vec<String>,
    pub enforcement_frequency: Duration,
    pub violation_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyValidation {
    pub validation_rules: Vec<String>,
    pub syntax_checking: bool,
    pub semantic_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyVersioning {
    pub version_control: bool,
    pub version_history: Vec<String>,
    pub rollback_capability: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyInheritance {
    pub inheritance_enabled: bool,
    pub inheritance_rules: Vec<String>,
    pub override_policies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchemaDefinition {
    pub schema_name: String,
    pub schema_version: String,
    pub schema_properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationRule {
    pub rule_name: String,
    pub rule_expression: String,
    pub error_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchemaRelationships {
    pub relationship_definitions: HashMap<String, String>,
    pub dependency_graph: HashMap<String, Vec<String>>,
    pub constraint_enforcement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchemaVersioning {
    pub versioning_strategy: String,
    pub backward_compatibility: bool,
    pub migration_support: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchemaEvolution {
    pub evolution_policies: Vec<String>,
    pub change_impact_analysis: bool,
    pub automated_migration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationValidation {
    pub validation_enabled: bool,
    pub validation_rules: Vec<String>,
    pub validation_severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationDeployment {
    pub deployment_strategy: String,
    pub rollback_capability: bool,
    pub deployment_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationMonitoring {
    pub monitoring_enabled: bool,
    pub change_detection: bool,
    pub drift_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationBackup {
    pub backup_frequency: Duration,
    pub backup_retention: Duration,
    pub backup_encryption: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationSynchronization {
    pub sync_enabled: bool,
    pub sync_frequency: Duration,
    pub conflict_resolution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DynamicConfiguration {
    pub dynamic_updates: bool,
    pub update_mechanisms: Vec<String>,
    pub validation_on_update: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceStandard {
    pub standard_name: String,
    pub standard_version: String,
    pub compliance_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CertificationRequirements {
    pub certification_body: String,
    pub certification_criteria: Vec<String>,
    pub renewal_frequency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AssessmentProcedures {
    pub assessment_frequency: Duration,
    pub assessment_criteria: Vec<String>,
    pub assessment_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FrameworkMapping {
    pub control_mappings: HashMap<String, Vec<String>>,
    pub requirement_mappings: HashMap<String, String>,
    pub gap_analysis: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditRequirements {
    pub audit_frequency: Duration,
    pub audit_scope: Vec<String>,
    pub audit_evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceMonitoring {
    pub monitoring_frequency: Duration,
    pub monitoring_scope: Vec<String>,
    pub automated_monitoring: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ViolationHandling {
    pub violation_detection: bool,
    pub escalation_procedures: Vec<String>,
    pub remediation_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceReporting {
    pub reporting_frequency: Duration,
    pub report_recipients: Vec<String>,
    pub report_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RemediationProcedures {
    pub remediation_steps: Vec<String>,
    pub remediation_timeline: Duration,
    pub verification_procedures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DelegationRule {
    pub delegation_scope: String,
    pub delegation_authority: String,
    pub delegation_limits: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OverrideMechanism {
    pub override_conditions: Vec<String>,
    pub override_authority: String,
    pub override_process: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FederationMember {
    pub member_id: String,
    pub member_authority: String,
    pub member_responsibilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CoordinationMechanisms {
    pub coordination_protocols: Vec<String>,
    pub communication_channels: Vec<String>,
    pub coordination_frequency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConflictResolution {
    pub resolution_procedures: Vec<String>,
    pub escalation_hierarchy: Vec<String>,
    pub arbitration_mechanisms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutonomousUnit {
    pub unit_id: String,
    pub unit_scope: String,
    pub autonomous_decisions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CoordinationProtocols {
    pub protocol_definitions: Vec<String>,
    pub protocol_enforcement: bool,
    pub protocol_monitoring: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsensusMechanisms {
    pub consensus_algorithms: Vec<String>,
    pub consensus_thresholds: HashMap<String, f64>,
    pub consensus_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HierarchicalComponent {
    pub component_id: String,
    pub component_level: u32,
    pub component_authority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntegrationStrategies {
    pub integration_approaches: Vec<String>,
    pub integration_protocols: Vec<String>,
    pub integration_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GovernanceInterfaces {
    pub interface_definitions: Vec<String>,
    pub interface_protocols: Vec<String>,
    pub interface_security: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DecisionMaking {
    pub decision_processes: Vec<String>,
    pub decision_criteria: Vec<String>,
    pub decision_authority: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccountabilityMatrix {
    pub responsibility_assignments: HashMap<String, Vec<String>>,
    pub accountability_measures: Vec<String>,
    pub performance_indicators: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApprovalWorkflows {
    pub workflow_definitions: HashMap<String, String>,
    pub approval_hierarchies: Vec<String>,
    pub workflow_automation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GovernanceRoles {
    pub role_definitions: HashMap<String, String>,
    pub role_responsibilities: HashMap<String, Vec<String>>,
    pub role_authorities: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EscalationProcedures {
    pub escalation_triggers: Vec<String>,
    pub escalation_paths: HashMap<String, Vec<String>>,
    pub escalation_timeouts: HashMap<String, Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalyticsFramework {
    pub analytics_tools: Vec<String>,
    pub data_collection: String,
    pub analysis_methods: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyMetrics {
    pub metric_definitions: HashMap<String, String>,
    pub measurement_frequency: Duration,
    pub metric_targets: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EffectivenessAnalysis {
    pub effectiveness_criteria: Vec<String>,
    pub analysis_methods: Vec<String>,
    pub improvement_recommendations: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceAnalytics {
    pub compliance_metrics: HashMap<String, f64>,
    pub trend_analysis: bool,
    pub predictive_modeling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrendAnalysis {
    pub trend_detection: bool,
    pub trend_forecasting: bool,
    pub trend_reporting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PredictiveAnalytics {
    pub prediction_models: Vec<String>,
    pub prediction_accuracy: f64,
    pub model_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportingDashboard {
    pub dashboard_components: Vec<String>,
    pub real_time_updates: bool,
    pub interactive_features: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditPolicy {
    pub policy_name: String,
    pub audit_scope: Vec<String>,
    pub audit_frequency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditTrails {
    pub trail_configuration: String,
    pub data_retention: Duration,
    pub trail_integrity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditProcedures {
    pub procedure_definitions: Vec<String>,
    pub procedure_automation: bool,
    pub procedure_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditReporting {
    pub report_templates: Vec<String>,
    pub reporting_schedule: String,
    pub report_distribution: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditAutomation {
    pub automated_audits: bool,
    pub automation_tools: Vec<String>,
    pub automation_coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExternalAuditIntegration {
    pub integration_enabled: bool,
    pub external_auditors: Vec<String>,
    pub integration_protocols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecycleStage {
    pub stage_name: String,
    pub stage_criteria: Vec<String>,
    pub stage_activities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransitionManagement {
    pub transition_procedures: Vec<String>,
    pub transition_validation: bool,
    pub rollback_procedures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeprecationPolicies {
    pub deprecation_criteria: Vec<String>,
    pub deprecation_timeline: Duration,
    pub migration_support: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MigrationStrategies {
    pub migration_approaches: Vec<String>,
    pub migration_validation: bool,
    pub data_migration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecycleGovernance {
    pub governance_processes: Vec<String>,
    pub approval_requirements: Vec<String>,
    pub stakeholder_involvement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EndOfLifeProcedures {
    pub eol_criteria: Vec<String>,
    pub data_archival: bool,
    pub cleanup_procedures: Vec<String>,
}

// Additional supporting types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeWindow {
    pub start_time: String,
    pub end_time: String,
    pub days_of_week: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecurringPattern {
    pub pattern_type: String,
    pub pattern_frequency: Duration,
    pub pattern_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExceptionPeriod {
    pub exception_start: Instant,
    pub exception_end: Instant,
    pub exception_reason: String,
}