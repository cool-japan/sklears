use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::{Duration, SystemTime};
use std::net::IpAddr;
use serde::{Serialize, Deserialize};

// ================================================================================================
// Core Security Analysis Types
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysisConfig {
    pub analysis_depth: AnalysisDepth,
    pub scope_definitions: Vec<ScopeDefinition>,
    pub risk_tolerance: RiskTolerance,
    pub compliance_requirements: Vec<String>,
    pub reporting_frequency: Duration,
    pub automated_remediation: bool,
    pub real_time_monitoring: bool,
    pub threat_intelligence_enabled: bool,
    pub vulnerability_scanning_enabled: bool,
    pub penetration_testing_enabled: bool,
    pub social_engineering_testing_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnalysisDepth {
    Surface,      // Basic security checks
    Standard,     // Comprehensive analysis
    Deep,         // Advanced techniques
    Exhaustive,   // Maximum coverage
    Custom(u8),   // Custom depth level
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeDefinition {
    pub scope_id: String,
    pub scope_name: String,
    pub included_components: Vec<String>,
    pub excluded_components: Vec<String>,
    pub analysis_priorities: Vec<AnalysisPriority>,
    pub resource_constraints: ResourceConstraints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisPriority {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_analysis_time: Duration,
    pub cpu_limit_percentage: f64,
    pub memory_limit_mb: u64,
    pub network_bandwidth_limit: u64,
    pub concurrent_operations_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskTolerance {
    pub overall_risk_appetite: RiskAppetite,
    pub category_tolerances: HashMap<RiskCategory, f64>,
    pub acceptable_vulnerability_levels: HashMap<VulnerabilitySeverity, u32>,
    pub maximum_exposure_duration: Duration,
    pub business_impact_thresholds: BusinessImpactThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskAppetite {
    Minimal,     // Risk-averse organization
    Conservative, // Cautious approach
    Moderate,    // Balanced risk/reward
    Aggressive,  // Higher risk tolerance
    Extreme,     // Maximum risk acceptance
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum RiskCategory {
    Operational,
    Financial,
    Strategic,
    Compliance,
    Reputational,
    Technical,
    Legal,
    Environmental,
}

// ================================================================================================
// Vulnerability and Threat Types
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum VulnerabilitySeverity {
    Critical,    // CVSS 9.0-10.0
    High,        // CVSS 7.0-8.9
    Medium,      // CVSS 4.0-6.9
    Low,         // CVSS 0.1-3.9
    Informational, // CVSS 0.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityDetails {
    pub vulnerability_id: String,
    pub cve_id: Option<String>,
    pub title: String,
    pub description: String,
    pub severity: VulnerabilitySeverity,
    pub cvss_score: Option<f64>,
    pub cvss_vector: Option<String>,
    pub cwe_id: Option<String>,
    pub affected_systems: Vec<String>,
    pub exploit_availability: ExploitAvailability,
    pub remediation_guidance: Vec<RemediationStep>,
    pub discovery_date: SystemTime,
    pub last_updated: SystemTime,
    pub status: VulnerabilityStatus,
    pub tags: Vec<String>,
    pub references: Vec<VulnerabilityReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExploitAvailability {
    None,           // No known exploits
    Proof_of_Concept, // PoC available
    Functional,     // Functional exploit exists
    High,           // Weaponized exploit
    Unknown,        // Status unknown
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilityStatus {
    New,            // Recently discovered
    Confirmed,      // Verified vulnerability
    In_Progress,    // Remediation in progress
    Resolved,       // Fixed/mitigated
    Accepted,       // Risk accepted
    False_Positive, // Not a real vulnerability
    Duplicate,      // Duplicate of another finding
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationStep {
    pub step_id: String,
    pub description: String,
    pub priority: AnalysisPriority,
    pub estimated_effort: Duration,
    pub required_skills: Vec<String>,
    pub dependencies: Vec<String>,
    pub validation_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityReference {
    pub reference_type: ReferenceType,
    pub url: String,
    pub description: String,
    pub publication_date: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReferenceType {
    Advisory,
    Vendor,
    Third_Party,
    Research,
    News,
    Blog,
    Social_Media,
    Technical_Analysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatActor {
    pub actor_id: String,
    pub name: String,
    pub aliases: Vec<String>,
    pub actor_type: ThreatActorType,
    pub sophistication_level: SophisticationLevel,
    pub motivations: Vec<ThreatMotivation>,
    pub target_sectors: Vec<String>,
    pub target_geographies: Vec<String>,
    pub attack_patterns: Vec<String>,
    pub tools_and_techniques: Vec<String>,
    pub attribution_confidence: f64,
    pub first_observed: SystemTime,
    pub last_activity: SystemTime,
    pub active_campaigns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatActorType {
    Nation_State,
    Criminal_Organization,
    Hacktivist,
    Insider_Threat,
    Script_Kiddie,
    Competitor,
    Terrorist,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SophisticationLevel {
    Minimal,      // Basic tools and techniques
    Intermediate, // Moderate skill and resources
    Advanced,     // High skill and significant resources
    Expert,       // State-level capabilities
    Unknown,      // Cannot be determined
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatMotivation {
    Financial_Gain,
    Espionage,
    Sabotage,
    Ideology,
    Revenge,
    Notoriety,
    Testing_Skills,
    Unknown,
}

// ================================================================================================
// Risk Assessment Types
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub assessment_id: String,
    pub assessment_date: SystemTime,
    pub scope: String,
    pub methodology: RiskMethodology,
    pub risk_items: Vec<RiskItem>,
    pub overall_risk_level: RiskLevel,
    pub risk_score: f64,
    pub confidence_level: f64,
    pub recommendations: Vec<RiskRecommendation>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
    pub residual_risk: f64,
    pub next_assessment_date: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskMethodology {
    Qualitative,
    Quantitative,
    Semi_Quantitative,
    NIST_SP800_30,
    ISO_27005,
    OCTAVE,
    FAIR,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskItem {
    pub risk_id: String,
    pub title: String,
    pub description: String,
    pub category: RiskCategory,
    pub threat_sources: Vec<String>,
    pub vulnerabilities: Vec<String>,
    pub likelihood: LikelihoodLevel,
    pub impact: ImpactLevel,
    pub risk_level: RiskLevel,
    pub risk_score: f64,
    pub existing_controls: Vec<String>,
    pub control_effectiveness: f64,
    pub residual_risk_score: f64,
    pub business_impact: BusinessImpactAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum LikelihoodLevel {
    Very_Low,    // 0-10%
    Low,         // 11-30%
    Medium,      // 31-70%
    High,        // 71-90%
    Very_High,   // 91-100%
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImpactLevel {
    Negligible,  // Minimal impact
    Minor,       // Limited impact
    Moderate,    // Significant impact
    Major,       // Severe impact
    Catastrophic, // Extreme impact
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Very_Low,
    Low,
    Medium,
    High,
    Very_High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpactAssessment {
    pub financial_impact: FinancialImpact,
    pub operational_impact: OperationalImpact,
    pub reputational_impact: ReputationalImpact,
    pub legal_impact: LegalImpact,
    pub customer_impact: CustomerImpact,
    pub competitive_impact: CompetitiveImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialImpact {
    pub direct_costs: f64,
    pub indirect_costs: f64,
    pub revenue_loss: f64,
    pub regulatory_fines: f64,
    pub legal_costs: f64,
    pub recovery_costs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalImpact {
    pub service_disruption_duration: Duration,
    pub affected_business_processes: Vec<String>,
    pub productivity_loss_percentage: f64,
    pub recovery_time_objective: Duration,
    pub recovery_point_objective: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationalImpact {
    pub brand_damage_score: f64,
    pub customer_trust_impact: f64,
    pub media_attention_level: MediaAttentionLevel,
    pub social_media_sentiment: SentimentLevel,
    pub stakeholder_confidence_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediaAttentionLevel {
    None,
    Local,
    Regional,
    National,
    International,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SentimentLevel {
    Very_Negative,
    Negative,
    Neutral,
    Positive,
    Very_Positive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalImpact {
    pub regulatory_violations: Vec<String>,
    pub potential_lawsuits: u32,
    pub compliance_breach_severity: ComplianceBreachSeverity,
    pub data_protection_violations: Vec<String>,
    pub contractual_breach_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceBreachSeverity {
    Minor,
    Moderate,
    Significant,
    Major,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerImpact {
    pub affected_customer_count: u64,
    pub customer_data_exposed: bool,
    pub service_availability_impact: f64,
    pub customer_satisfaction_impact: f64,
    pub churn_risk_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitiveImpact {
    pub competitive_advantage_loss: f64,
    pub intellectual_property_exposure: bool,
    pub market_share_impact: f64,
    pub innovation_capability_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpactThresholds {
    pub financial_impact_thresholds: HashMap<ImpactLevel, f64>,
    pub operational_disruption_thresholds: HashMap<ImpactLevel, Duration>,
    pub customer_impact_thresholds: HashMap<ImpactLevel, u64>,
    pub reputational_impact_thresholds: HashMap<ImpactLevel, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskRecommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub priority: AnalysisPriority,
    pub category: RecommendationCategory,
    pub implementation_effort: ImplementationEffort,
    pub expected_risk_reduction: f64,
    pub cost_estimate: f64,
    pub timeline: Duration,
    pub dependencies: Vec<String>,
    pub success_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Technical_Control,
    Administrative_Control,
    Physical_Control,
    Process_Improvement,
    Training,
    Technology_Investment,
    Policy_Update,
    Governance_Change,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationEffort {
    pub effort_level: EffortLevel,
    pub required_skills: Vec<String>,
    pub estimated_hours: u32,
    pub resource_requirements: Vec<String>,
    pub external_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Minimal,      // < 8 hours
    Low,          // 8-40 hours
    Medium,       // 40-200 hours
    High,         // 200-1000 hours
    Very_High,    // > 1000 hours
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    pub strategy_id: String,
    pub name: String,
    pub description: String,
    pub strategy_type: MitigationType,
    pub effectiveness_rating: f64,
    pub implementation_cost: f64,
    pub ongoing_cost: f64,
    pub implementation_time: Duration,
    pub maintenance_requirements: Vec<String>,
    pub success_metrics: Vec<String>,
    pub rollback_plan: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MitigationType {
    Accept,       // Accept the risk
    Avoid,        // Eliminate the risk
    Mitigate,     // Reduce the risk
    Transfer,     // Transfer the risk to others
    Monitor,      // Monitor and reassess
}

// ================================================================================================
// Compliance and Audit Types
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFramework {
    pub framework_id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub issuing_organization: String,
    pub effective_date: SystemTime,
    pub geographic_scope: Vec<String>,
    pub industry_applicability: Vec<String>,
    pub framework_type: ComplianceFrameworkType,
    pub requirements: Vec<ComplianceRequirement>,
    pub controls: Vec<ComplianceControl>,
    pub assessment_procedures: Vec<AssessmentProcedure>,
    pub certification_available: bool,
    pub mandatory_compliance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceFrameworkType {
    Regulatory,    // Government regulation
    Standard,      // Industry standard
    Best_Practice, // Best practice framework
    Internal,      // Internal policy
    Contractual,   // Contract requirement
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    pub requirement_id: String,
    pub title: String,
    pub description: String,
    pub obligation_level: ObligationLevel,
    pub applicable_roles: Vec<String>,
    pub implementation_guidance: String,
    pub assessment_criteria: Vec<String>,
    pub evidence_requirements: Vec<String>,
    pub related_controls: Vec<String>,
    pub penalties_for_non_compliance: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObligationLevel {
    Mandatory,    // Must be implemented
    Recommended,  // Should be implemented
    Optional,     // May be implemented
    Conditional,  // Required under certain conditions
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceControl {
    pub control_id: String,
    pub title: String,
    pub description: String,
    pub control_type: ControlType,
    pub implementation_guidance: String,
    pub testing_procedures: Vec<String>,
    pub frequency: TestingFrequency,
    pub responsible_parties: Vec<String>,
    pub dependencies: Vec<String>,
    pub compensating_controls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlType {
    Preventive,   // Prevent incidents
    Detective,    // Detect incidents
    Corrective,   // Correct incidents
    Deterrent,    // Deter incidents
    Recovery,     // Recover from incidents
    Compensating, // Alternative control
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestingFrequency {
    Continuous,   // Real-time monitoring
    Daily,        // Daily testing
    Weekly,       // Weekly testing
    Monthly,      // Monthly testing
    Quarterly,    // Quarterly testing
    Semi_Annual,  // Twice yearly
    Annual,       // Yearly testing
    Ad_Hoc,       // As needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentProcedure {
    pub procedure_id: String,
    pub name: String,
    pub description: String,
    pub assessment_type: AssessmentType,
    pub scope: String,
    pub methodology: String,
    pub required_evidence: Vec<String>,
    pub assessment_criteria: Vec<String>,
    pub pass_fail_criteria: String,
    pub estimated_duration: Duration,
    pub required_expertise: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssessmentType {
    Self_Assessment,
    Internal_Audit,
    External_Audit,
    Third_Party_Assessment,
    Continuous_Monitoring,
    Penetration_Test,
    Vulnerability_Assessment,
    Code_Review,
}

// ================================================================================================
// Security Event and Incident Types
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub event_id: String,
    pub timestamp: SystemTime,
    pub event_type: SecurityEventType,
    pub severity: EventSeverity,
    pub source_system: String,
    pub source_ip: Option<IpAddr>,
    pub user_context: Option<UserContext>,
    pub event_details: HashMap<String, String>,
    pub raw_log_data: Option<String>,
    pub correlation_id: Option<String>,
    pub event_signature: Option<String>,
    pub false_positive_likelihood: f64,
    pub investigation_status: InvestigationStatus,
    pub escalation_level: EscalationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    Authentication_Failure,
    Authorization_Violation,
    Data_Access_Anomaly,
    Network_Intrusion_Attempt,
    Malware_Detection,
    Data_Exfiltration_Attempt,
    System_Compromise,
    Policy_Violation,
    Configuration_Change,
    Privilege_Escalation,
    Lateral_Movement,
    Command_And_Control,
    Data_Destruction,
    Service_Disruption,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSeverity {
    Informational, // Normal activity
    Low,           // Minor security relevance
    Medium,        // Moderate security concern
    High,          // Significant security incident
    Critical,      // Major security breach
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RiskSeverity {
    Low,      // Low risk level
    Medium,   // Medium risk level
    High,     // High risk level
    Critical, // Critical risk level
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ThreatSeverity {
    Low,      // Low threat level
    Medium,   // Medium threat level
    High,     // High threat level
    Critical, // Critical threat level
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub user_id: String,
    pub username: String,
    pub roles: Vec<String>,
    pub privileges: Vec<String>,
    pub session_id: Option<String>,
    pub authentication_method: String,
    pub source_location: Option<String>,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvestigationStatus {
    New,           // Recently detected
    Assigned,      // Assigned to analyst
    In_Progress,   // Under investigation
    Escalated,     // Escalated to higher level
    Resolved,      // Investigation complete
    Closed,        // Case closed
    False_Positive, // Determined not malicious
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationLevel {
    Level_1,       // SOC analyst
    Level_2,       // Senior analyst
    Level_3,       // Security engineer
    Management,    // Management notification
    Executive,     // Executive notification
    External,      // External parties (law enforcement, etc.)
}

// ================================================================================================
// Security Metrics and Reporting Types
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetric {
    pub metric_id: String,
    pub metric_name: String,
    pub metric_type: SecurityMetricType,
    pub description: String,
    pub calculation_method: String,
    pub data_sources: Vec<String>,
    pub collection_frequency: Duration,
    pub target_value: Option<MetricValue>,
    pub current_value: MetricValue,
    pub historical_values: Vec<TimestampedMetricValue>,
    pub trend: TrendDirection,
    pub thresholds: Vec<MetricThreshold>,
    pub business_context: String,
    pub stakeholders: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityMetricType {
    Vulnerability_Metric,
    Threat_Metric,
    Risk_Metric,
    Compliance_Metric,
    Incident_Metric,
    Performance_Metric,
    Effectiveness_Metric,
    Maturity_Metric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Integer(i64),
    Float(f64),
    Percentage(f64),
    Boolean(bool),
    Text(String),
    Duration(Duration),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedMetricValue {
    pub timestamp: SystemTime,
    pub value: MetricValue,
    pub quality_score: f64,
    pub collection_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,     // Getting better
    Deteriorating, // Getting worse
    Stable,        // No significant change
    Volatile,      // Highly variable
    Unknown,       // Cannot determine
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricThreshold {
    pub threshold_name: String,
    pub threshold_type: ThresholdType,
    pub threshold_value: MetricValue,
    pub action_required: String,
    pub notification_recipients: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdType {
    Upper_Warning,   // Warn if value exceeds threshold
    Upper_Critical,  // Critical if value exceeds threshold
    Lower_Warning,   // Warn if value below threshold
    Lower_Critical,  // Critical if value below threshold
    Range_Warning,   // Warn if value outside range
    Range_Critical,  // Critical if value outside range
}

// ================================================================================================
// Security Configuration Types
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfiguration {
    pub config_id: String,
    pub config_name: String,
    pub config_version: String,
    pub description: String,
    pub applicable_systems: Vec<String>,
    pub configuration_items: Vec<ConfigurationItem>,
    pub baseline_requirements: Vec<BaselineRequirement>,
    pub hardening_guidelines: Vec<HardeningGuideline>,
    pub monitoring_requirements: Vec<MonitoringRequirement>,
    pub change_control_requirements: Vec<ChangeControlRequirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationItem {
    pub item_id: String,
    pub item_name: String,
    pub item_type: ConfigurationItemType,
    pub required_value: ConfigurationValue,
    pub current_value: Option<ConfigurationValue>,
    pub compliance_status: ConfigurationComplianceStatus,
    pub risk_level_if_non_compliant: RiskLevel,
    pub remediation_guidance: String,
    pub validation_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationItemType {
    Registry_Key,
    File_Permission,
    Service_Configuration,
    Network_Setting,
    User_Account_Setting,
    Password_Policy,
    Audit_Policy,
    Firewall_Rule,
    Encryption_Setting,
    Certificate_Configuration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationValue {
    String(String),
    Integer(i64),
    Boolean(bool),
    List(Vec<String>),
    Range(i64, i64),
    RegexPattern(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationComplianceStatus {
    Compliant,
    Non_Compliant,
    Partially_Compliant,
    Unknown,
    Not_Applicable,
    Exempted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRequirement {
    pub requirement_id: String,
    pub title: String,
    pub description: String,
    pub category: BaselineCategory,
    pub mandatory: bool,
    pub applicable_platforms: Vec<String>,
    pub implementation_guidance: String,
    pub validation_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineCategory {
    Access_Control,
    Audit_And_Logging,
    Authentication,
    Encryption,
    Network_Security,
    System_Hardening,
    Data_Protection,
    Incident_Response,
    Backup_And_Recovery,
    Physical_Security,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardeningGuideline {
    pub guideline_id: String,
    pub title: String,
    pub description: String,
    pub security_benefit: String,
    pub implementation_steps: Vec<String>,
    pub potential_impacts: Vec<String>,
    pub rollback_procedure: String,
    pub verification_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringRequirement {
    pub requirement_id: String,
    pub monitored_component: String,
    pub monitoring_frequency: Duration,
    pub alert_conditions: Vec<AlertCondition>,
    pub data_retention_period: Duration,
    pub responsible_team: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub condition_name: String,
    pub condition_logic: String,
    pub severity: EventSeverity,
    pub notification_channels: Vec<String>,
    pub escalation_rules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeControlRequirement {
    pub requirement_id: String,
    pub change_type: ChangeType,
    pub approval_required: bool,
    pub required_approvers: Vec<String>,
    pub testing_requirements: Vec<String>,
    pub documentation_requirements: Vec<String>,
    pub rollback_requirements: Vec<String>,
    pub post_change_verification: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Emergency,    // Immediate implementation required
    Standard,     // Normal change process
    Minor,        // Pre-approved change
    Major,        // High-impact change
}

// ================================================================================================
// Error Types and Results
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityAnalysisError {
    ConfigurationError(String),
    DataAccessError(String),
    ValidationError(String),
    AnalysisError(String),
    ReportingError(String),
    IntegrationError(String),
    PermissionError(String),
    ResourceError(String),
    TimeoutError(String),
    NetworkError(String),
    AuthenticationError(String),
    AuthorizationError(String),
    CryptographicError(String),
    ComplianceError(String),
    AuditError(String),
}

impl std::fmt::Display for SecurityAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityAnalysisError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            SecurityAnalysisError::DataAccessError(msg) => write!(f, "Data access error: {}", msg),
            SecurityAnalysisError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            SecurityAnalysisError::AnalysisError(msg) => write!(f, "Analysis error: {}", msg),
            SecurityAnalysisError::ReportingError(msg) => write!(f, "Reporting error: {}", msg),
            SecurityAnalysisError::IntegrationError(msg) => write!(f, "Integration error: {}", msg),
            SecurityAnalysisError::PermissionError(msg) => write!(f, "Permission error: {}", msg),
            SecurityAnalysisError::ResourceError(msg) => write!(f, "Resource error: {}", msg),
            SecurityAnalysisError::TimeoutError(msg) => write!(f, "Timeout error: {}", msg),
            SecurityAnalysisError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            SecurityAnalysisError::AuthenticationError(msg) => write!(f, "Authentication error: {}", msg),
            SecurityAnalysisError::AuthorizationError(msg) => write!(f, "Authorization error: {}", msg),
            SecurityAnalysisError::CryptographicError(msg) => write!(f, "Cryptographic error: {}", msg),
            SecurityAnalysisError::ComplianceError(msg) => write!(f, "Compliance error: {}", msg),
            SecurityAnalysisError::AuditError(msg) => write!(f, "Audit error: {}", msg),
        }
    }
}

impl std::error::Error for SecurityAnalysisError {}

pub type SecurityAnalysisResult<T> = Result<T, SecurityAnalysisError>;

// ================================================================================================
// Default Implementations
// ================================================================================================

impl Default for SecurityAnalysisConfig {
    fn default() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Standard,
            scope_definitions: Vec::new(),
            risk_tolerance: RiskTolerance::default(),
            compliance_requirements: vec!["NIST".to_string(), "ISO27001".to_string()],
            reporting_frequency: Duration::from_secs(86400), // Daily
            automated_remediation: false,
            real_time_monitoring: true,
            threat_intelligence_enabled: true,
            vulnerability_scanning_enabled: true,
            penetration_testing_enabled: false,
            social_engineering_testing_enabled: false,
        }
    }
}

impl Default for RiskTolerance {
    fn default() -> Self {
        let mut category_tolerances = HashMap::new();
        category_tolerances.insert(RiskCategory::Technical, 0.7);
        category_tolerances.insert(RiskCategory::Operational, 0.6);
        category_tolerances.insert(RiskCategory::Financial, 0.5);
        category_tolerances.insert(RiskCategory::Compliance, 0.3);

        let mut acceptable_vulnerability_levels = HashMap::new();
        acceptable_vulnerability_levels.insert(VulnerabilitySeverity::Critical, 0);
        acceptable_vulnerability_levels.insert(VulnerabilitySeverity::High, 5);
        acceptable_vulnerability_levels.insert(VulnerabilitySeverity::Medium, 20);
        acceptable_vulnerability_levels.insert(VulnerabilitySeverity::Low, 100);

        let mut financial_thresholds = HashMap::new();
        financial_thresholds.insert(ImpactLevel::Minor, 10000.0);
        financial_thresholds.insert(ImpactLevel::Moderate, 100000.0);
        financial_thresholds.insert(ImpactLevel::Major, 1000000.0);
        financial_thresholds.insert(ImpactLevel::Catastrophic, 10000000.0);

        let mut operational_thresholds = HashMap::new();
        operational_thresholds.insert(ImpactLevel::Minor, Duration::from_secs(3600));
        operational_thresholds.insert(ImpactLevel::Moderate, Duration::from_secs(28800));
        operational_thresholds.insert(ImpactLevel::Major, Duration::from_secs(86400));
        operational_thresholds.insert(ImpactLevel::Catastrophic, Duration::from_secs(259200));

        let mut customer_thresholds = HashMap::new();
        customer_thresholds.insert(ImpactLevel::Minor, 1000);
        customer_thresholds.insert(ImpactLevel::Moderate, 10000);
        customer_thresholds.insert(ImpactLevel::Major, 100000);
        customer_thresholds.insert(ImpactLevel::Catastrophic, 1000000);

        let mut reputational_thresholds = HashMap::new();
        reputational_thresholds.insert(ImpactLevel::Minor, 0.1);
        reputational_thresholds.insert(ImpactLevel::Moderate, 0.3);
        reputational_thresholds.insert(ImpactLevel::Major, 0.6);
        reputational_thresholds.insert(ImpactLevel::Catastrophic, 0.9);

        Self {
            overall_risk_appetite: RiskAppetite::Moderate,
            category_tolerances,
            acceptable_vulnerability_levels,
            maximum_exposure_duration: Duration::from_secs(86400 * 30), // 30 days
            business_impact_thresholds: BusinessImpactThresholds {
                financial_impact_thresholds: financial_thresholds,
                operational_disruption_thresholds: operational_thresholds,
                customer_impact_thresholds: customer_thresholds,
                reputational_impact_thresholds: reputational_thresholds,
            },
        }
    }
}

// ================================================================================================
// Utility Functions and Trait Implementations
// ================================================================================================

impl VulnerabilitySeverity {
    pub fn from_cvss_score(score: f64) -> Self {
        match score {
            s if s >= 9.0 => VulnerabilitySeverity::Critical,
            s if s >= 7.0 => VulnerabilitySeverity::High,
            s if s >= 4.0 => VulnerabilitySeverity::Medium,
            s if s > 0.0 => VulnerabilitySeverity::Low,
            _ => VulnerabilitySeverity::Informational,
        }
    }

    pub fn to_numeric_value(&self) -> u8 {
        match self {
            VulnerabilitySeverity::Critical => 5,
            VulnerabilitySeverity::High => 4,
            VulnerabilitySeverity::Medium => 3,
            VulnerabilitySeverity::Low => 2,
            VulnerabilitySeverity::Informational => 1,
        }
    }
}

impl RiskLevel {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 9.0 => RiskLevel::Critical,
            s if s >= 7.0 => RiskLevel::Very_High,
            s if s >= 5.0 => RiskLevel::High,
            s if s >= 3.0 => RiskLevel::Medium,
            s if s >= 1.0 => RiskLevel::Low,
            _ => RiskLevel::Very_Low,
        }
    }

    pub fn to_numeric_value(&self) -> u8 {
        match self {
            RiskLevel::Critical => 6,
            RiskLevel::Very_High => 5,
            RiskLevel::High => 4,
            RiskLevel::Medium => 3,
            RiskLevel::Low => 2,
            RiskLevel::Very_Low => 1,
        }
    }
}

impl LikelihoodLevel {
    pub fn to_probability(&self) -> f64 {
        match self {
            LikelihoodLevel::Very_Low => 0.05,
            LikelihoodLevel::Low => 0.2,
            LikelihoodLevel::Medium => 0.5,
            LikelihoodLevel::High => 0.8,
            LikelihoodLevel::Very_High => 0.95,
        }
    }

    pub fn from_probability(prob: f64) -> Self {
        match prob {
            p if p <= 0.1 => LikelihoodLevel::Very_Low,
            p if p <= 0.3 => LikelihoodLevel::Low,
            p if p <= 0.7 => LikelihoodLevel::Medium,
            p if p <= 0.9 => LikelihoodLevel::High,
            _ => LikelihoodLevel::Very_High,
        }
    }
}

impl ImpactLevel {
    pub fn to_numeric_value(&self) -> u8 {
        match self {
            ImpactLevel::Negligible => 1,
            ImpactLevel::Minor => 2,
            ImpactLevel::Moderate => 3,
            ImpactLevel::Major => 4,
            ImpactLevel::Catastrophic => 5,
        }
    }
}

// ================================================================================================
// Constants and Static Values
// ================================================================================================

pub const DEFAULT_ANALYSIS_TIMEOUT: Duration = Duration::from_secs(3600); // 1 hour
pub const DEFAULT_REPORT_RETENTION: Duration = Duration::from_secs(86400 * 365); // 1 year
pub const DEFAULT_VULNERABILITY_SCAN_FREQUENCY: Duration = Duration::from_secs(86400); // Daily
pub const DEFAULT_THREAT_INTELLIGENCE_REFRESH: Duration = Duration::from_secs(3600); // Hourly
pub const DEFAULT_RISK_ASSESSMENT_FREQUENCY: Duration = Duration::from_secs(86400 * 30); // Monthly
pub const DEFAULT_COMPLIANCE_CHECK_FREQUENCY: Duration = Duration::from_secs(86400 * 7); // Weekly

pub const CVSS_V3_MAX_SCORE: f64 = 10.0;
pub const CVSS_V3_MIN_SCORE: f64 = 0.0;
pub const RISK_SCORE_MAX: f64 = 10.0;
pub const RISK_SCORE_MIN: f64 = 0.0;
pub const CONFIDENCE_SCORE_MAX: f64 = 1.0;
pub const CONFIDENCE_SCORE_MIN: f64 = 0.0;

// Common vulnerability categories based on CWE
pub const COMMON_VULNERABILITY_CATEGORIES: &[&str] = &[
    "CWE-79: Cross-site Scripting",
    "CWE-89: SQL Injection",
    "CWE-200: Information Exposure",
    "CWE-264: Permissions, Privileges, and Access Controls",
    "CWE-287: Improper Authentication",
    "CWE-352: Cross-Site Request Forgery",
    "CWE-434: Unrestricted Upload of File with Dangerous Type",
    "CWE-502: Deserialization of Untrusted Data",
    "CWE-601: URL Redirection to Untrusted Site",
    "CWE-798: Use of Hard-coded Credentials",
];

// Common attack patterns based on MITRE ATT&CK
pub const COMMON_ATTACK_PATTERNS: &[&str] = &[
    "T1566: Phishing",
    "T1078: Valid Accounts",
    "T1190: Exploit Public-Facing Application",
    "T1059: Command and Scripting Interpreter",
    "T1055: Process Injection",
    "T1021: Remote Services",
    "T1003: OS Credential Dumping",
    "T1005: Data from Local System",
    "T1041: Exfiltration Over C2 Channel",
    "T1486: Data Encrypted for Impact",
];

// Security framework mappings
pub const NIST_CSF_FUNCTIONS: &[&str] = &["Identify", "Protect", "Detect", "Respond", "Recover"];
pub const ISO27001_DOMAINS: &[&str] = &[
    "Information Security Policies",
    "Organization of Information Security",
    "Human Resource Security",
    "Asset Management",
    "Access Control",
    "Cryptography",
    "Physical and Environmental Security",
    "Operations Security",
    "Communications Security",
    "System Acquisition, Development and Maintenance",
    "Supplier Relationships",
    "Information Security Incident Management",
    "Information Security Aspects of Business Continuity Management",
    "Compliance",
];

/// Utility function to create a new vulnerability details instance
pub fn create_vulnerability_details(
    id: String,
    title: String,
    description: String,
    severity: VulnerabilitySeverity,
) -> VulnerabilityDetails {
    VulnerabilityDetails {
        vulnerability_id: id,
        cve_id: None,
        title,
        description,
        severity,
        cvss_score: None,
        cvss_vector: None,
        cwe_id: None,
        affected_systems: Vec::new(),
        exploit_availability: ExploitAvailability::Unknown,
        remediation_guidance: Vec::new(),
        discovery_date: SystemTime::now(),
        last_updated: SystemTime::now(),
        status: VulnerabilityStatus::New,
        tags: Vec::new(),
        references: Vec::new(),
    }
}

/// Utility function to create a new risk assessment instance
pub fn create_risk_assessment(
    id: String,
    scope: String,
    methodology: RiskMethodology,
) -> RiskAssessment {
    RiskAssessment {
        assessment_id: id,
        assessment_date: SystemTime::now(),
        scope,
        methodology,
        risk_items: Vec::new(),
        overall_risk_level: RiskLevel::Medium,
        risk_score: 5.0,
        confidence_level: 0.8,
        recommendations: Vec::new(),
        mitigation_strategies: Vec::new(),
        residual_risk: 3.0,
        next_assessment_date: SystemTime::now() + Duration::from_secs(86400 * 90), // 90 days
    }
}

/// Utility function to create a new security event instance
pub fn create_security_event(
    event_type: SecurityEventType,
    severity: EventSeverity,
    source_system: String,
) -> SecurityEvent {
    SecurityEvent {
        event_id: format!("evt_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).expect("duration_since should succeed").as_secs()),
        timestamp: SystemTime::now(),
        event_type,
        severity,
        source_system,
        source_ip: None,
        user_context: None,
        event_details: HashMap::new(),
        raw_log_data: None,
        correlation_id: None,
        event_signature: None,
        false_positive_likelihood: 0.1,
        investigation_status: InvestigationStatus::New,
        escalation_level: EscalationLevel::Level_1,
    }
}