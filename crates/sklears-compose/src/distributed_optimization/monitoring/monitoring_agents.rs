use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringAgentManager {
    pub active_agents: HashMap<String, MonitoringAgent>,
    pub agent_registry: AgentRegistry,
    pub lifecycle_management: LifecycleManagement,
    pub plugin_management: PluginManagement,
    pub agent_coordination: AgentCoordination,
    pub performance_monitoring: PerformanceMonitoring,
    pub security_framework: SecurityFramework,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringAgent {
    pub agent_id: String,
    pub agent_type: AgentType,
    pub agent_configuration: AgentConfiguration,
    pub current_state: AgentState,
    pub capabilities: Vec<AgentCapability>,
    pub performance_metrics: AgentMetrics,
    pub communication_interface: CommunicationInterface,
    pub resource_requirements: ResourceRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    MetricsCollector {
        collection_scope: CollectionScope,
        collection_frequency: Duration,
        data_formats: Vec<DataFormat>,
    },
    HealthMonitor {
        monitoring_targets: Vec<String>,
        health_checks: Vec<HealthCheck>,
        alert_thresholds: AlertThresholds,
    },
    LogAnalyzer {
        log_sources: Vec<LogSource>,
        analysis_patterns: Vec<AnalysisPattern>,
        real_time_processing: bool,
    },
    PerformanceTracker {
        tracked_metrics: Vec<String>,
        baseline_computation: BaselineComputation,
        anomaly_detection: bool,
    },
    SecurityMonitor {
        security_domains: Vec<SecurityDomain>,
        threat_detection: ThreatDetection,
        incident_response: IncidentResponse,
    },
    NetworkMonitor {
        network_segments: Vec<String>,
        traffic_analysis: TrafficAnalysis,
        topology_mapping: bool,
    },
    ResourceMonitor {
        resource_types: Vec<ResourceType>,
        utilization_tracking: UtilizationTracking,
        capacity_planning: CapacityPlanning,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRegistry {
    pub registered_agents: HashMap<String, AgentRegistration>,
    pub discovery_mechanisms: Vec<DiscoveryMechanism>,
    pub registration_policies: RegistrationPolicies,
    pub agent_dependencies: AgentDependencies,
    pub registry_persistence: RegistryPersistence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRegistration {
    pub registration_id: String,
    pub agent_metadata: AgentMetadata,
    pub registration_timestamp: Instant,
    pub registration_status: RegistrationStatus,
    pub health_status: HealthStatus,
    pub last_heartbeat: Instant,
    pub registration_details: RegistrationDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleManagement {
    pub lifecycle_policies: HashMap<String, LifecyclePolicy>,
    pub deployment_strategies: DeploymentStrategies,
    pub upgrade_management: UpgradeManagement,
    pub termination_procedures: TerminationProcedures,
    pub state_transitions: StateTransitions,
    pub lifecycle_monitoring: LifecycleMonitoring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecyclePolicy {
    AutomaticManagement {
        auto_start: bool,
        auto_restart: bool,
        auto_scaling: AutoScalingConfig,
    },
    ManualManagement {
        approval_required: bool,
        authorized_operators: Vec<String>,
    },
    ScheduledManagement {
        start_schedule: String,
        stop_schedule: String,
        maintenance_windows: Vec<MaintenanceWindow>,
    },
    ConditionalManagement {
        start_conditions: Vec<Condition>,
        stop_conditions: Vec<Condition>,
        health_conditions: Vec<Condition>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManagement {
    pub plugin_registry: PluginRegistry,
    pub plugin_loader: PluginLoader,
    pub plugin_isolation: PluginIsolation,
    pub plugin_communication: PluginCommunication,
    pub plugin_security: PluginSecurity,
    pub plugin_lifecycle: PluginLifecycle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRegistry {
    pub available_plugins: HashMap<String, PluginDescriptor>,
    pub plugin_categories: HashMap<String, Vec<String>>,
    pub plugin_dependencies: PluginDependencies,
    pub plugin_compatibility: PluginCompatibility,
    pub plugin_versioning: PluginVersioning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDescriptor {
    pub plugin_id: String,
    pub plugin_name: String,
    pub plugin_version: String,
    pub plugin_type: PluginType,
    pub capabilities: Vec<PluginCapability>,
    pub configuration_schema: ConfigurationSchema,
    pub resource_requirements: PluginResourceRequirements,
    pub security_permissions: SecurityPermissions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginType {
    DataCollector {
        supported_sources: Vec<String>,
        output_formats: Vec<String>,
    },
    DataProcessor {
        processing_functions: Vec<String>,
        transformation_capabilities: Vec<String>,
    },
    Analyzer {
        analysis_algorithms: Vec<String>,
        result_formats: Vec<String>,
    },
    Visualizer {
        chart_types: Vec<String>,
        export_formats: Vec<String>,
    },
    Notifier {
        notification_channels: Vec<String>,
        message_formats: Vec<String>,
    },
    Integrator {
        integration_protocols: Vec<String>,
        supported_systems: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCoordination {
    pub coordination_protocols: Vec<CoordinationProtocol>,
    pub message_routing: MessageRouting,
    pub synchronization_mechanisms: SynchronizationMechanisms,
    pub conflict_resolution: ConflictResolution,
    pub load_balancing: LoadBalancing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationProtocol {
    MasterSlave {
        master_agent: String,
        coordination_frequency: Duration,
    },
    PeerToPeer {
        consensus_algorithm: ConsensusAlgorithm,
        network_topology: NetworkTopology,
    },
    Hierarchical {
        hierarchy_levels: Vec<HierarchyLevel>,
        delegation_policies: DelegationPolicies,
    },
    EventDriven {
        event_types: Vec<String>,
        subscription_model: SubscriptionModel,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoring {
    pub performance_metrics: Vec<PerformanceMetric>,
    pub benchmarking_framework: BenchmarkingFramework,
    pub performance_analysis: PerformanceAnalysis,
    pub optimization_recommendations: OptimizationRecommendations,
    pub performance_reporting: PerformanceReporting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFramework {
    pub authentication_mechanisms: Vec<AuthenticationMechanism>,
    pub authorization_policies: AuthorizationPolicies,
    pub encryption_protocols: EncryptionProtocols,
    pub audit_logging: AuditLogging,
    pub security_monitoring: SecurityMonitoring,
    pub threat_mitigation: ThreatMitigation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfiguration {
    pub configuration_parameters: HashMap<String, ConfigurationValue>,
    pub runtime_configuration: RuntimeConfiguration,
    pub configuration_validation: ConfigurationValidation,
    pub configuration_versioning: ConfigurationVersioning,
    pub dynamic_reconfiguration: DynamicReconfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentState {
    Initializing,
    Running,
    Paused,
    Stopping,
    Stopped,
    Error { error_code: String, error_message: String },
    Upgrading,
    Maintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentCapability {
    DataCollection { sources: Vec<String> },
    DataProcessing { algorithms: Vec<String> },
    RealTimeAnalysis { latency_requirements: Duration },
    HistoricalAnalysis { retention_period: Duration },
    Alerting { alert_types: Vec<String> },
    Reporting { report_formats: Vec<String> },
    Integration { protocols: Vec<String> },
    Scaling { scaling_modes: Vec<String> },
}

impl Default for MonitoringAgentManager {
    fn default() -> Self {
        Self {
            active_agents: HashMap::new(),
            agent_registry: AgentRegistry::default(),
            lifecycle_management: LifecycleManagement::default(),
            plugin_management: PluginManagement::default(),
            agent_coordination: AgentCoordination::default(),
            performance_monitoring: PerformanceMonitoring::default(),
            security_framework: SecurityFramework::default(),
        }
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self {
            registered_agents: HashMap::new(),
            discovery_mechanisms: vec![DiscoveryMechanism::default()],
            registration_policies: RegistrationPolicies::default(),
            agent_dependencies: AgentDependencies::default(),
            registry_persistence: RegistryPersistence::default(),
        }
    }
}

impl Default for LifecycleManagement {
    fn default() -> Self {
        Self {
            lifecycle_policies: HashMap::new(),
            deployment_strategies: DeploymentStrategies::default(),
            upgrade_management: UpgradeManagement::default(),
            termination_procedures: TerminationProcedures::default(),
            state_transitions: StateTransitions::default(),
            lifecycle_monitoring: LifecycleMonitoring::default(),
        }
    }
}

impl Default for PluginManagement {
    fn default() -> Self {
        Self {
            plugin_registry: PluginRegistry::default(),
            plugin_loader: PluginLoader::default(),
            plugin_isolation: PluginIsolation::default(),
            plugin_communication: PluginCommunication::default(),
            plugin_security: PluginSecurity::default(),
            plugin_lifecycle: PluginLifecycle::default(),
        }
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self {
            available_plugins: HashMap::new(),
            plugin_categories: HashMap::new(),
            plugin_dependencies: PluginDependencies::default(),
            plugin_compatibility: PluginCompatibility::default(),
            plugin_versioning: PluginVersioning::default(),
        }
    }
}

impl Default for AgentCoordination {
    fn default() -> Self {
        Self {
            coordination_protocols: vec![CoordinationProtocol::EventDriven {
                event_types: vec!["status_change".to_string(), "metric_update".to_string()],
                subscription_model: SubscriptionModel::default(),
            }],
            message_routing: MessageRouting::default(),
            synchronization_mechanisms: SynchronizationMechanisms::default(),
            conflict_resolution: ConflictResolution::default(),
            load_balancing: LoadBalancing::default(),
        }
    }
}

impl Default for PerformanceMonitoring {
    fn default() -> Self {
        Self {
            performance_metrics: Vec::new(),
            benchmarking_framework: BenchmarkingFramework::default(),
            performance_analysis: PerformanceAnalysis::default(),
            optimization_recommendations: OptimizationRecommendations::default(),
            performance_reporting: PerformanceReporting::default(),
        }
    }
}

impl Default for SecurityFramework {
    fn default() -> Self {
        Self {
            authentication_mechanisms: vec![AuthenticationMechanism::default()],
            authorization_policies: AuthorizationPolicies::default(),
            encryption_protocols: EncryptionProtocols::default(),
            audit_logging: AuditLogging::default(),
            security_monitoring: SecurityMonitoring::default(),
            threat_mitigation: ThreatMitigation::default(),
        }
    }
}

// Supporting types with Default implementations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionScope {
    pub scope_type: String,
    pub target_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataFormat {
    pub format_name: String,
    pub format_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthCheck {
    pub check_name: String,
    pub check_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertThresholds {
    pub warning_threshold: f64,
    pub critical_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LogSource {
    pub source_id: String,
    pub source_type: String,
    pub source_location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalysisPattern {
    pub pattern_name: String,
    pub pattern_expression: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BaselineComputation {
    pub computation_method: String,
    pub computation_window: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityDomain {
    pub domain_name: String,
    pub security_policies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThreatDetection {
    pub detection_algorithms: Vec<String>,
    pub threat_signatures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IncidentResponse {
    pub response_procedures: Vec<String>,
    pub escalation_rules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrafficAnalysis {
    pub analysis_methods: Vec<String>,
    pub sampling_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceType {
    pub type_name: String,
    pub measurement_units: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UtilizationTracking {
    pub tracking_granularity: Duration,
    pub historical_retention: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CapacityPlanning {
    pub planning_horizon: Duration,
    pub growth_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiscoveryMechanism {
    pub mechanism_type: String,
    pub discovery_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistrationPolicies {
    pub auto_registration: bool,
    pub approval_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentDependencies {
    pub dependency_graph: HashMap<String, Vec<String>>,
    pub resolution_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryPersistence {
    pub persistence_backend: String,
    pub backup_frequency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RegistrationStatus {
    #[default]
    Pending,
    Approved,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum HealthStatus {
    #[default]
    Healthy,
    Warning,
    Critical,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistrationDetails {
    pub registration_metadata: HashMap<String, String>,
    pub contact_information: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutoScalingConfig {
    pub min_instances: u32,
    pub max_instances: u32,
    pub scaling_triggers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaintenanceWindow {
    pub start_time: String,
    pub duration: Duration,
    pub recurrence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Condition {
    pub condition_type: String,
    pub condition_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginCapability {
    pub capability_name: String,
    pub capability_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationSchema {
    pub schema_definition: String,
    pub validation_rules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginResourceRequirements {
    pub cpu_requirements: f64,
    pub memory_requirements: usize,
    pub storage_requirements: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityPermissions {
    pub required_permissions: Vec<String>,
    pub permission_scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsensusAlgorithm {
    pub algorithm_type: String,
    pub algorithm_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkTopology {
    pub topology_type: String,
    pub node_connections: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HierarchyLevel {
    pub level_name: String,
    pub level_agents: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DelegationPolicies {
    pub delegation_rules: Vec<String>,
    pub authority_matrix: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubscriptionModel {
    pub subscription_type: String,
    pub subscription_policies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetric {
    pub metric_name: String,
    pub measurement_unit: String,
    pub collection_frequency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthenticationMechanism {
    pub mechanism_type: String,
    pub configuration: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationValue {
    pub value_type: String,
    pub value: String,
    pub is_sensitive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentMetrics {
    pub performance_metrics: HashMap<String, f64>,
    pub resource_utilization: HashMap<String, f64>,
    pub operational_metrics: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommunicationInterface {
    pub interface_type: String,
    pub communication_protocols: Vec<String>,
    pub message_formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_mb: usize,
    pub disk_space_mb: usize,
    pub network_bandwidth: f64,
}

// Additional default implementations for remaining types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeploymentStrategies {
    pub strategy_type: String,
    pub deployment_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpgradeManagement {
    pub upgrade_strategy: String,
    pub rollback_capability: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TerminationProcedures {
    pub graceful_shutdown: bool,
    pub shutdown_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateTransitions {
    pub allowed_transitions: HashMap<String, Vec<String>>,
    pub transition_validations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecycleMonitoring {
    pub monitoring_frequency: Duration,
    pub lifecycle_events: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginDependencies {
    pub dependency_graph: HashMap<String, Vec<String>>,
    pub version_constraints: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginCompatibility {
    pub compatibility_matrix: HashMap<String, Vec<String>>,
    pub compatibility_testing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginVersioning {
    pub versioning_scheme: String,
    pub version_history: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginLoader {
    pub loading_strategy: String,
    pub isolation_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginIsolation {
    pub isolation_mechanisms: Vec<String>,
    pub resource_limits: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginCommunication {
    pub communication_bus: String,
    pub message_routing: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginSecurity {
    pub security_policies: Vec<String>,
    pub permission_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginLifecycle {
    pub lifecycle_hooks: Vec<String>,
    pub lifecycle_management: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessageRouting {
    pub routing_algorithm: String,
    pub routing_table: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SynchronizationMechanisms {
    pub synchronization_protocols: Vec<String>,
    pub coordination_overhead: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConflictResolution {
    pub resolution_strategy: String,
    pub priority_rules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LoadBalancing {
    pub balancing_algorithm: String,
    pub load_distribution: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkingFramework {
    pub benchmark_suites: Vec<String>,
    pub performance_baselines: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceAnalysis {
    pub analysis_methods: Vec<String>,
    pub performance_trends: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationRecommendations {
    pub recommendation_engine: String,
    pub optimization_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceReporting {
    pub report_formats: Vec<String>,
    pub reporting_frequency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthorizationPolicies {
    pub policy_engine: String,
    pub access_control_rules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EncryptionProtocols {
    pub encryption_algorithms: Vec<String>,
    pub key_management: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditLogging {
    pub log_level: String,
    pub audit_events: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityMonitoring {
    pub monitoring_scope: Vec<String>,
    pub threat_intelligence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThreatMitigation {
    pub mitigation_strategies: Vec<String>,
    pub response_automation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeConfiguration {
    pub runtime_parameters: HashMap<String, String>,
    pub configuration_refresh: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationValidation {
    pub validation_rules: Vec<String>,
    pub validation_on_update: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationVersioning {
    pub version_tracking: bool,
    pub rollback_capability: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DynamicReconfiguration {
    pub hot_reload: bool,
    pub configuration_propagation: String,
}