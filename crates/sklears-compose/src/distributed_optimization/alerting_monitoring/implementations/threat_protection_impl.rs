use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Threat protection configuration
/// Handles intrusion detection, data loss prevention, malware protection, and threat intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatProtectionConfig {
    /// Intrusion detection system configuration
    pub intrusion_detection: IntrusionDetectionConfig,
    /// Data loss prevention configuration
    pub data_loss_prevention: DataLossPreventionConfig,
    /// Malware protection configuration
    pub malware_protection: MalwareProtectionConfig,
    /// Threat intelligence configuration
    pub threat_intelligence: ThreatIntelligenceConfig,
}

/// Intrusion detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrusionDetectionConfig {
    /// Enable intrusion detection
    pub enabled: bool,
    /// Detection methods to use
    pub detection_methods: Vec<DetectionMethod>,
    /// Response actions for detected intrusions
    pub response_actions: Vec<ResponseAction>,
    /// Alert thresholds for different severity levels
    pub alert_thresholds: AlertThresholds,
}

/// Detection methods for intrusion detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionMethod {
    SignatureBased,
    AnomalyBased,
    BehaviorBased,
    HeuristicBased,
    MachineLearning,
}

/// Response actions for detected threats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseAction {
    /// Generate alert
    Alert,
    /// Block the threat
    Block,
    /// Quarantine the threat
    Quarantine,
    /// Disconnect the source
    Disconnect,
    /// Log the event
    Log,
    /// Custom response action
    Custom(String),
}

/// Alert thresholds for different severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Low severity threshold
    pub low_threshold: f64,
    /// Medium severity threshold
    pub medium_threshold: f64,
    /// High severity threshold
    pub high_threshold: f64,
    /// Critical severity threshold
    pub critical_threshold: f64,
}

/// Data loss prevention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLossPreventionConfig {
    /// Enable data loss prevention
    pub enabled: bool,
    /// Content inspection configuration
    pub content_inspection: ContentInspectionConfig,
    /// Policy enforcement configuration
    pub policy_enforcement: PolicyEnforcementConfig,
    /// Incident response configuration
    pub incident_response: IncidentResponseConfig,
}

/// Content inspection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentInspectionConfig {
    /// Depth of content inspection
    pub inspection_depth: InspectionDepth,
    /// Enable file type filtering
    pub file_type_filtering: bool,
    /// Pattern matching rules
    pub pattern_matching: Vec<String>,
    /// Enable machine learning detection
    pub machine_learning_detection: bool,
}

/// Inspection depth levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InspectionDepth {
    /// Shallow inspection (headers, metadata)
    Shallow,
    /// Deep inspection (partial content)
    Deep,
    /// Full inspection (complete content)
    Full,
}

/// Policy enforcement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEnforcementConfig {
    /// Enforcement mode
    pub enforcement_mode: EnforcementMode,
    /// Actions to take on policy violations
    pub violation_actions: Vec<ViolationAction>,
    /// Policy exemptions
    pub exemptions: Vec<PolicyExemption>,
}

/// Enforcement modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementMode {
    /// Monitor only (no blocking)
    Monitor,
    /// Enforce policies (block violations)
    Enforce,
    /// Hybrid mode (monitor some, enforce others)
    Hybrid,
}

/// Actions for policy violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationAction {
    /// Block the action
    Block,
    /// Quarantine the content
    Quarantine,
    /// Encrypt the content
    Encrypt,
    /// Redact sensitive information
    Redact,
    /// Generate alert
    Alert,
    /// Log the violation
    Log,
}

/// Policy exemption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyExemption {
    /// Exemption name
    pub exemption_name: String,
    /// Conditions for exemption
    pub conditions: Vec<ExemptionCondition>,
    /// Exemption expiration time
    pub expiration: Option<SystemTime>,
    /// Justification for exemption
    pub justification: String,
}

/// Exemption condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExemptionCondition {
    /// Attribute to check
    pub attribute: String,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Value to compare against
    pub value: serde_json::Value,
}

/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Greater than or equal to
    GreaterThanOrEqual,
    /// Less than or equal to
    LessThanOrEqual,
    /// Contains
    Contains,
    /// Starts with
    StartsWith,
    /// Ends with
    EndsWith,
    /// Regular expression match
    Regex,
}

/// Incident response configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentResponseConfig {
    /// Enable automated response
    pub automated_response: bool,
    /// Escalation procedures
    pub escalation_procedures: Vec<EscalationProcedure>,
    /// Notification workflows
    pub notification_workflows: Vec<NotificationWorkflow>,
    /// Enable forensic data collection
    pub forensic_collection: bool,
}

/// Escalation procedure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationProcedure {
    /// Severity level for escalation
    pub severity_level: String,
    /// Delay before escalation
    pub escalation_delay: Duration,
    /// Escalation targets
    pub escalation_targets: Vec<String>,
    /// Required actions during escalation
    pub required_actions: Vec<String>,
}

/// Notification workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationWorkflow {
    /// Workflow name
    pub workflow_name: String,
    /// Trigger conditions for workflow
    pub trigger_conditions: Vec<TriggerCondition>,
    /// Notification steps
    pub notification_steps: Vec<NotificationStep>,
}

/// Trigger condition for workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerCondition {
    /// Condition type
    pub condition_type: String,
    /// Condition parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Notification step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationStep {
    /// Step name
    pub step_name: String,
    /// Notification method (email, SMS, etc.)
    pub notification_method: String,
    /// Recipients list
    pub recipients: Vec<String>,
    /// Message template
    pub message_template: String,
    /// Delay before sending
    pub delay: Option<Duration>,
}

/// Malware protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MalwareProtectionConfig {
    /// Enable malware protection
    pub enabled: bool,
    /// Scanning engines
    pub scanning_engines: Vec<ScanningEngine>,
    /// Enable real-time protection
    pub real_time_protection: bool,
    /// Quarantine configuration
    pub quarantine_config: QuarantineConfig,
}

/// Scanning engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanningEngine {
    /// Engine name
    pub engine_name: String,
    /// Engine type
    pub engine_type: ScanningEngineType,
    /// Update frequency for signatures
    pub update_frequency: Duration,
    /// Scanning depth
    pub scan_depth: ScanDepth,
}

/// Scanning engine types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScanningEngineType {
    /// Signature-based scanning
    SignatureBased,
    /// Heuristic-based scanning
    HeuristicBased,
    /// Behavior-based scanning
    BehaviorBased,
    /// Cloud-based scanning
    CloudBased,
    /// Hybrid scanning approach
    Hybrid,
}

/// Scan depth options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScanDepth {
    /// Quick scan (basic files)
    Quick,
    /// Full scan (all files)
    Full,
    /// Deep scan (with unpacking)
    Deep,
    /// Custom scan (specified file types)
    Custom(Vec<String>),
}

/// Quarantine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineConfig {
    /// Quarantine storage location
    pub quarantine_location: String,
    /// Enable encryption for quarantined files
    pub encryption_enabled: bool,
    /// Retention period for quarantined files
    pub retention_period: Duration,
    /// Access restrictions for quarantine area
    pub access_restrictions: Vec<String>,
}

/// Threat intelligence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIntelligenceConfig {
    /// Enable threat intelligence
    pub enabled: bool,
    /// Intelligence sources
    pub intelligence_sources: Vec<IntelligenceSource>,
    /// Integration configuration
    pub integration_config: ThreatIntegrationConfig,
    /// Sharing configuration
    pub sharing_config: ThreatSharingConfig,
}

/// Intelligence source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceSource {
    /// Source name
    pub source_name: String,
    /// Source type
    pub source_type: IntelligenceSourceType,
    /// Update frequency
    pub update_frequency: Duration,
    /// Trust level for this source
    pub trust_level: TrustLevel,
}

/// Intelligence source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntelligenceSourceType {
    /// Commercial threat intelligence
    Commercial,
    /// Government sources
    Government,
    /// Open source intelligence
    OpenSource,
    /// Community sources
    Community,
    /// Internal intelligence
    Internal,
}

/// Trust levels for sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Low trust
    Low,
    /// Medium trust
    Medium,
    /// High trust
    High,
    /// Critical trust
    Critical,
}

/// Threat integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIntegrationConfig {
    /// Enable automated threat data ingestion
    pub automated_ingestion: bool,
    /// Correlation rules for threat data
    pub correlation_rules: Vec<CorrelationRule>,
    /// Enable threat data enrichment
    pub enrichment_enabled: bool,
    /// Scoring algorithm for threats
    pub scoring_algorithm: ScoringAlgorithm,
}

/// Correlation rule for threat analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRule {
    /// Rule name
    pub rule_name: String,
    /// Pattern matching rules
    pub patterns: Vec<String>,
    /// Confidence threshold for correlation
    pub confidence_threshold: f64,
    /// Actions to take on correlation match
    pub actions: Vec<String>,
}

/// Scoring algorithms for threat assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringAlgorithm {
    /// Simple scoring algorithm
    Simple,
    /// Weighted scoring algorithm
    Weighted,
    /// Bayesian scoring algorithm
    Bayesian,
    /// Machine learning-based scoring
    MachineLearning,
    /// Custom scoring algorithm
    Custom(String),
}

/// Threat sharing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatSharingConfig {
    /// Enable threat intelligence sharing
    pub sharing_enabled: bool,
    /// Anonymization level for shared data
    pub anonymization_level: AnonymizationLevel,
    /// Sharing partners
    pub sharing_partners: Vec<SharingPartner>,
    /// Sharing protocols
    pub sharing_protocols: Vec<SharingProtocol>,
}

/// Anonymization levels for threat data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnonymizationLevel {
    /// No anonymization
    None,
    /// Partial anonymization
    Partial,
    /// Full anonymization
    Full,
    /// Differential privacy
    Differential,
}

/// Sharing partner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingPartner {
    /// Partner name
    pub partner_name: String,
    /// Partner type
    pub partner_type: PartnerType,
    /// Trust level for partner
    pub trust_level: TrustLevel,
    /// Sharing agreement details
    pub sharing_agreement: SharingAgreement,
}

/// Partner types for threat sharing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartnerType {
    /// Government partner
    Government,
    /// Commercial partner
    Commercial,
    /// Academic partner
    Academic,
    /// Non-profit partner
    NonProfit,
    /// International partner
    International,
}

/// Sharing agreement details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingAgreement {
    /// Agreement type
    pub agreement_type: AgreementType,
    /// Data categories covered
    pub data_categories: Vec<String>,
    /// Usage restrictions
    pub usage_restrictions: Vec<String>,
    /// Data retention limits
    pub retention_limits: Option<Duration>,
}

/// Agreement types for data sharing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgreementType {
    /// Bilateral agreement
    Bilateral,
    /// Multilateral agreement
    Multilateral,
    /// Community agreement
    Community,
    /// Commercial agreement
    Commercial,
}

/// Sharing protocols for threat intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingProtocol {
    /// Structured Threat Information eXpression
    STIX,
    /// Trusted Automated eXchange of Indicator Information
    TAXII,
    /// Incident Object Description Exchange Format
    IODEF,
    /// Open Indicators of Compromise
    OpenIOC,
    /// Custom sharing protocol
    Custom(String),
}

// Default implementations
impl Default for ThreatProtectionConfig {
    fn default() -> Self {
        Self {
            intrusion_detection: IntrusionDetectionConfig::default(),
            data_loss_prevention: DataLossPreventionConfig::default(),
            malware_protection: MalwareProtectionConfig::default(),
            threat_intelligence: ThreatIntelligenceConfig::default(),
        }
    }
}

impl Default for IntrusionDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detection_methods: vec![
                DetectionMethod::SignatureBased,
                DetectionMethod::AnomalyBased,
            ],
            response_actions: vec![
                ResponseAction::Alert,
                ResponseAction::Log,
            ],
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            low_threshold: 0.3,
            medium_threshold: 0.6,
            high_threshold: 0.8,
            critical_threshold: 0.95,
        }
    }
}

impl Default for DataLossPreventionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            content_inspection: ContentInspectionConfig::default(),
            policy_enforcement: PolicyEnforcementConfig::default(),
            incident_response: IncidentResponseConfig::default(),
        }
    }
}

impl Default for ContentInspectionConfig {
    fn default() -> Self {
        Self {
            inspection_depth: InspectionDepth::Deep,
            file_type_filtering: true,
            pattern_matching: Vec::new(),
            machine_learning_detection: false,
        }
    }
}

impl Default for PolicyEnforcementConfig {
    fn default() -> Self {
        Self {
            enforcement_mode: EnforcementMode::Monitor,
            violation_actions: vec![
                ViolationAction::Alert,
                ViolationAction::Log,
            ],
            exemptions: Vec::new(),
        }
    }
}

impl Default for IncidentResponseConfig {
    fn default() -> Self {
        Self {
            automated_response: false,
            escalation_procedures: Vec::new(),
            notification_workflows: Vec::new(),
            forensic_collection: false,
        }
    }
}

impl Default for MalwareProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scanning_engines: vec![
                ScanningEngine {
                    engine_name: "Default".to_string(),
                    engine_type: ScanningEngineType::SignatureBased,
                    update_frequency: Duration::from_secs(3600), // 1 hour
                    scan_depth: ScanDepth::Full,
                }
            ],
            real_time_protection: true,
            quarantine_config: QuarantineConfig::default(),
        }
    }
}

impl Default for QuarantineConfig {
    fn default() -> Self {
        Self {
            quarantine_location: "/var/quarantine".to_string(),
            encryption_enabled: true,
            retention_period: Duration::from_secs(30 * 24 * 3600), // 30 days
            access_restrictions: vec!["admin".to_string()],
        }
    }
}

impl Default for ThreatIntelligenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intelligence_sources: Vec::new(),
            integration_config: ThreatIntegrationConfig::default(),
            sharing_config: ThreatSharingConfig::default(),
        }
    }
}

impl Default for ThreatIntegrationConfig {
    fn default() -> Self {
        Self {
            automated_ingestion: false,
            correlation_rules: Vec::new(),
            enrichment_enabled: false,
            scoring_algorithm: ScoringAlgorithm::Simple,
        }
    }
}

impl Default for ThreatSharingConfig {
    fn default() -> Self {
        Self {
            sharing_enabled: false,
            anonymization_level: AnonymizationLevel::Full,
            sharing_partners: Vec::new(),
            sharing_protocols: vec![SharingProtocol::STIX],
        }
    }
}

impl ThreatProtectionConfig {
    /// Create a new threat protection configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable intrusion detection with specific methods
    pub fn enable_intrusion_detection(&mut self, methods: Vec<DetectionMethod>) {
        self.intrusion_detection.enabled = true;
        self.intrusion_detection.detection_methods = methods;
    }

    /// Configure data loss prevention
    pub fn configure_dlp(&mut self, enforcement_mode: EnforcementMode, actions: Vec<ViolationAction>) {
        self.data_loss_prevention.enabled = true;
        self.data_loss_prevention.policy_enforcement.enforcement_mode = enforcement_mode;
        self.data_loss_prevention.policy_enforcement.violation_actions = actions;
    }

    /// Enable malware protection with real-time scanning
    pub fn enable_malware_protection(&mut self, real_time: bool, engines: Vec<ScanningEngine>) {
        self.malware_protection.enabled = true;
        self.malware_protection.real_time_protection = real_time;
        self.malware_protection.scanning_engines = engines;
    }

    /// Configure threat intelligence sharing
    pub fn configure_threat_intelligence(&mut self,
        sources: Vec<IntelligenceSource>,
        sharing_enabled: bool,
        anonymization: AnonymizationLevel) {
        self.threat_intelligence.enabled = true;
        self.threat_intelligence.intelligence_sources = sources;
        self.threat_intelligence.sharing_config.sharing_enabled = sharing_enabled;
        self.threat_intelligence.sharing_config.anonymization_level = anonymization;
    }

    /// Add threat intelligence source
    pub fn add_intelligence_source(&mut self, source: IntelligenceSource) {
        self.threat_intelligence.intelligence_sources.push(source);
    }

    /// Add sharing partner
    pub fn add_sharing_partner(&mut self, partner: SharingPartner) {
        self.threat_intelligence.sharing_config.sharing_partners.push(partner);
    }

    /// Set alert thresholds
    pub fn set_alert_thresholds(&mut self, thresholds: AlertThresholds) {
        self.intrusion_detection.alert_thresholds = thresholds;
    }

    /// Add correlation rule for threat analysis
    pub fn add_correlation_rule(&mut self, rule: CorrelationRule) {
        self.threat_intelligence.integration_config.correlation_rules.push(rule);
    }

    /// Enable automated incident response
    pub fn enable_automated_response(&mut self, enabled: bool) {
        self.data_loss_prevention.incident_response.automated_response = enabled;
    }

    /// Add notification workflow
    pub fn add_notification_workflow(&mut self, workflow: NotificationWorkflow) {
        self.data_loss_prevention.incident_response.notification_workflows.push(workflow);
    }

    /// Configure quarantine settings
    pub fn configure_quarantine(&mut self, config: QuarantineConfig) {
        self.malware_protection.quarantine_config = config;
    }

    /// Get security status summary
    pub fn get_security_status(&self) -> SecurityStatus {
        SecurityStatus {
            intrusion_detection_enabled: self.intrusion_detection.enabled,
            dlp_enabled: self.data_loss_prevention.enabled,
            malware_protection_enabled: self.malware_protection.enabled,
            threat_intelligence_enabled: self.threat_intelligence.enabled,
            real_time_protection: self.malware_protection.real_time_protection,
            sharing_enabled: self.threat_intelligence.sharing_config.sharing_enabled,
        }
    }
}

/// Security status summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStatus {
    /// Intrusion detection status
    pub intrusion_detection_enabled: bool,
    /// Data loss prevention status
    pub dlp_enabled: bool,
    /// Malware protection status
    pub malware_protection_enabled: bool,
    /// Threat intelligence status
    pub threat_intelligence_enabled: bool,
    /// Real-time protection status
    pub real_time_protection: bool,
    /// Threat sharing status
    pub sharing_enabled: bool,
}

// Helper functions for creating common configurations
impl ThreatProtectionConfig {
    /// Create a basic enterprise configuration
    pub fn enterprise_default() -> Self {
        let mut config = Self::default();

        // Enable comprehensive intrusion detection
        config.enable_intrusion_detection(vec![
            DetectionMethod::SignatureBased,
            DetectionMethod::AnomalyBased,
            DetectionMethod::BehaviorBased,
        ]);

        // Configure DLP in enforce mode
        config.configure_dlp(
            EnforcementMode::Enforce,
            vec![
                ViolationAction::Block,
                ViolationAction::Alert,
                ViolationAction::Log,
            ]
        );

        // Enable comprehensive malware protection
        config.enable_malware_protection(
            true,
            vec![
                ScanningEngine {
                    engine_name: "Primary".to_string(),
                    engine_type: ScanningEngineType::Hybrid,
                    update_frequency: Duration::from_secs(1800), // 30 minutes
                    scan_depth: ScanDepth::Deep,
                }
            ]
        );

        // Enable threat intelligence
        config.threat_intelligence.enabled = true;
        config.threat_intelligence.integration_config.automated_ingestion = true;
        config.threat_intelligence.integration_config.enrichment_enabled = true;

        config
    }

    /// Create a high-security configuration
    pub fn high_security() -> Self {
        let mut config = Self::enterprise_default();

        // Set strict alert thresholds
        config.set_alert_thresholds(AlertThresholds {
            low_threshold: 0.2,
            medium_threshold: 0.4,
            high_threshold: 0.6,
            critical_threshold: 0.8,
        });

        // Enable all detection methods
        config.intrusion_detection.detection_methods = vec![
            DetectionMethod::SignatureBased,
            DetectionMethod::AnomalyBased,
            DetectionMethod::BehaviorBased,
            DetectionMethod::HeuristicBased,
            DetectionMethod::MachineLearning,
        ];

        // Enable automated incident response
        config.enable_automated_response(true);

        // Configure full content inspection
        config.data_loss_prevention.content_inspection.inspection_depth = InspectionDepth::Full;
        config.data_loss_prevention.content_inspection.machine_learning_detection = true;

        config
    }
}