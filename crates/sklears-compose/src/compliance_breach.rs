//! Breach detection, incident response, and investigation management
//!
//! This module provides comprehensive breach management capabilities including
//! breach detection, impact assessment, investigation workflows, evidence
//! collection, and incident response for regulatory compliance.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::compliance_core::{RegulatoryFramework, SensitivityLevel};
use crate::compliance_consent::DataCategory;
use crate::compliance_governance::SensitiveDataType;
use crate::compliance_regulatory::ComplianceSeverity;

/// Breach detector
#[derive(Debug)]
pub struct BreachDetector {
    /// Detection rules
    pub detection_rules: Vec<BreachDetectionRule>,
    /// Detected breaches
    pub breaches: VecDeque<DataBreach>,
    /// Investigation cases
    pub investigations: HashMap<String, BreachInvestigation>,
    /// Configuration
    pub config: BreachDetectorConfig,
}

/// Breach detection rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachDetectionRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Detection pattern
    pub pattern: String,
    /// Rule severity
    pub severity: BreachSeverity,
    /// Threshold
    pub threshold: f64,
    /// Time window
    pub time_window: Duration,
    /// Rule enabled
    pub enabled: bool,
}

/// Breach severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BreachSeverity {
    /// Low severity breach
    Low = 1,
    /// Medium severity breach
    Medium = 2,
    /// High severity breach
    High = 3,
    /// Critical severity breach
    Critical = 4,
    /// Catastrophic breach
    Catastrophic = 5,
}

/// Data breach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBreach {
    /// Breach ID
    pub id: Uuid,
    /// Breach type
    pub breach_type: BreachType,
    /// Breach severity
    pub severity: BreachSeverity,
    /// Discovery date
    pub discovered_at: SystemTime,
    /// Occurrence date
    pub occurred_at: Option<SystemTime>,
    /// Affected data
    pub affected_data: AffectedData,
    /// Breach source
    pub source: BreachSource,
    /// Breach status
    pub status: BreachStatus,
    /// Notification required
    pub notification_required: bool,
    /// Notification deadline
    pub notification_deadline: Option<SystemTime>,
    /// Impact assessment
    pub impact_assessment: ImpactAssessment,
}

/// Breach types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreachType {
    /// Unauthorized access
    UnauthorizedAccess,
    /// Data exfiltration
    DataExfiltration,
    /// Data destruction
    DataDestruction,
    /// Data corruption
    DataCorruption,
    /// System compromise
    SystemCompromise,
    /// Insider threat
    InsiderThreat,
    /// Accidental disclosure
    AccidentalDisclosure,
    /// Third-party breach
    ThirdPartyBreach,
    /// Custom breach type
    Custom(String),
}

/// Affected data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedData {
    /// Number of data subjects
    pub data_subjects_count: usize,
    /// Data categories
    pub data_categories: HashSet<DataCategory>,
    /// Sensitive data types
    pub sensitive_types: HashSet<SensitiveDataType>,
    /// Geographic scope
    pub geographic_scope: HashSet<String>,
    /// Data volume estimate
    pub volume_estimate: DataVolumeEstimate,
}

/// Data volume estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataVolumeEstimate {
    /// Records count
    pub records: Option<usize>,
    /// File count
    pub files: Option<usize>,
    /// Data size in bytes
    pub size_bytes: Option<usize>,
    /// Confidence level
    pub confidence: f64,
}

/// Breach source
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreachSource {
    /// Internal source
    Internal,
    /// External source
    External,
    /// Third party
    ThirdParty,
    /// Unknown source
    Unknown,
    /// Custom source
    Custom(String),
}

/// Breach status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreachStatus {
    /// Detected
    Detected,
    /// Under investigation
    Investigation,
    /// Confirmed
    Confirmed,
    /// Contained
    Contained,
    /// Resolved
    Resolved,
    /// Closed
    Closed,
    /// False positive
    FalsePositive,
}

/// Impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    /// Business impact
    pub business_impact: BusinessImpact,
    /// Individual impact
    pub individual_impact: IndividualImpact,
    /// Regulatory impact
    pub regulatory_impact: RegulatoryImpact,
    /// Reputational impact
    pub reputational_impact: ReputationalImpact,
    /// Financial impact
    pub financial_impact: FinancialImpact,
}

/// Business impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    /// Service disruption
    pub service_disruption: bool,
    /// Operational impact
    pub operational_impact: ImpactLevel,
    /// Competitive advantage loss
    pub competitive_loss: bool,
    /// Business continuity impact
    pub continuity_impact: ImpactLevel,
}

/// Individual impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndividualImpact {
    /// Privacy impact
    pub privacy_impact: ImpactLevel,
    /// Identity theft risk
    pub identity_theft_risk: RiskLevel,
    /// Financial harm risk
    pub financial_harm_risk: RiskLevel,
    /// Emotional distress potential
    pub emotional_distress: ImpactLevel,
}

/// Regulatory impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryImpact {
    /// Applicable regulations
    pub applicable_regulations: HashSet<RegulatoryFramework>,
    /// Violation severity
    pub violation_severity: ComplianceSeverity,
    /// Potential fines
    pub potential_fines: Option<f64>,
    /// Investigation likelihood
    pub investigation_likelihood: RiskLevel,
}

/// Reputational impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationalImpact {
    /// Media attention risk
    pub media_attention: RiskLevel,
    /// Customer trust impact
    pub customer_trust: ImpactLevel,
    /// Brand value impact
    pub brand_impact: ImpactLevel,
    /// Social media impact
    pub social_media_impact: ImpactLevel,
}

/// Financial impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialImpact {
    /// Immediate costs
    pub immediate_costs: Option<f64>,
    /// Long-term costs
    pub long_term_costs: Option<f64>,
    /// Revenue impact
    pub revenue_impact: Option<f64>,
    /// Recovery costs
    pub recovery_costs: Option<f64>,
}

/// Impact levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Minimal impact
    Minimal = 1,
    /// Low impact
    Low = 2,
    /// Medium impact
    Medium = 3,
    /// High impact
    High = 4,
    /// Severe impact
    Severe = 5,
}

/// Risk levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Very low risk
    VeryLow = 1,
    /// Low risk
    Low = 2,
    /// Medium risk
    Medium = 3,
    /// High risk
    High = 4,
    /// Very high risk
    VeryHigh = 5,
}

/// Breach investigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachInvestigation {
    /// Investigation ID
    pub id: String,
    /// Breach ID
    pub breach_id: Uuid,
    /// Investigation status
    pub status: InvestigationStatus,
    /// Lead investigator
    pub lead_investigator: String,
    /// Investigation team
    pub team: Vec<String>,
    /// Start date
    pub started_at: SystemTime,
    /// End date
    pub ended_at: Option<SystemTime>,
    /// Investigation findings
    pub findings: Vec<InvestigationFinding>,
    /// Evidence collected
    pub evidence: Vec<Evidence>,
    /// Remediation actions
    pub remediation_actions: Vec<String>,
}

/// Investigation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvestigationStatus {
    /// Initiated
    Initiated,
    /// In progress
    InProgress,
    /// Suspended
    Suspended,
    /// Completed
    Completed,
    /// Cancelled
    Cancelled,
}

/// Investigation finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestigationFinding {
    /// Finding ID
    pub id: String,
    /// Finding description
    pub description: String,
    /// Evidence references
    pub evidence_refs: Vec<String>,
    /// Confidence level
    pub confidence: f64,
    /// Impact on breach assessment
    pub impact: String,
}

/// Evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Evidence ID
    pub id: String,
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Evidence description
    pub description: String,
    /// Collection date
    pub collected_at: SystemTime,
    /// Collector
    pub collector: String,
    /// Chain of custody
    pub chain_of_custody: Vec<CustodyEntry>,
    /// Evidence location
    pub location: String,
    /// Hash/checksum
    pub checksum: Option<String>,
}

/// Evidence types (reusing from consent module)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Digital evidence
    Digital,
    /// Physical evidence
    Physical,
    /// Witness statement
    Witness,
    /// System logs
    SystemLogs,
    /// Network traffic
    NetworkTraffic,
    /// Forensic image
    ForensicImage,
    /// Custom evidence type
    Custom(String),
}

/// Chain of custody entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustodyEntry {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Person
    pub person: String,
    /// Action taken
    pub action: String,
    /// Notes
    pub notes: Option<String>,
}

/// Breach detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachDetectorConfig {
    /// Enable real-time detection
    pub real_time_detection: bool,
    /// Detection sensitivity
    pub sensitivity: SensitivityLevel,
    /// Auto-escalation enabled
    pub auto_escalation: bool,
    /// Investigation timeout
    pub investigation_timeout: Duration,
}

impl Default for BreachDetectorConfig {
    fn default() -> Self {
        Self {
            real_time_detection: true,
            sensitivity: SensitivityLevel::Medium,
            auto_escalation: true,
            investigation_timeout: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
        }
    }
}

impl BreachDetector {
    /// Create a new breach detector
    pub fn new() -> Self {
        Self {
            detection_rules: Vec::new(),
            breaches: VecDeque::new(),
            investigations: HashMap::new(),
            config: BreachDetectorConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BreachDetectorConfig) -> Self {
        Self {
            detection_rules: Vec::new(),
            breaches: VecDeque::new(),
            investigations: HashMap::new(),
            config,
        }
    }

    /// Add detection rule
    pub fn add_detection_rule(&mut self, rule: BreachDetectionRule) {
        self.detection_rules.push(rule);
    }

    /// Remove detection rule
    pub fn remove_detection_rule(&mut self, rule_id: &str) -> bool {
        if let Some(pos) = self.detection_rules.iter().position(|r| r.id == rule_id) {
            self.detection_rules.remove(pos);
            true
        } else {
            false
        }
    }

    /// Report a breach
    pub fn report_breach(&mut self, breach: DataBreach) {
        // Check if notification is required based on breach characteristics
        let notification_required = self.requires_notification(&breach);
        let notification_deadline = if notification_required {
            Some(SystemTime::now() + Duration::from_secs(72 * 60 * 60)) // 72 hours for GDPR
        } else {
            None
        };

        let mut breach = breach;
        breach.notification_required = notification_required;
        breach.notification_deadline = notification_deadline;

        self.breaches.push_back(breach);

        // Auto-escalate if configured
        if self.config.auto_escalation {
            self.auto_escalate_breach(&breach);
        }
    }

    /// Check if breach requires notification
    fn requires_notification(&self, breach: &DataBreach) -> bool {
        // High risk to rights and freedoms of individuals
        match breach.severity {
            BreachSeverity::High | BreachSeverity::Critical | BreachSeverity::Catastrophic => true,
            _ => {
                // Check for sensitive data types
                breach.affected_data.sensitive_types.contains(&SensitiveDataType::Pii) ||
                breach.affected_data.sensitive_types.contains(&SensitiveDataType::Phi) ||
                breach.affected_data.data_subjects_count > 1000
            }
        }
    }

    /// Auto-escalate breach
    fn auto_escalate_breach(&mut self, breach: &DataBreach) {
        if breach.severity >= BreachSeverity::High {
            // Create investigation
            let investigation = BreachInvestigation {
                id: format!("inv-{}", Uuid::new_v4()),
                breach_id: breach.id,
                status: InvestigationStatus::Initiated,
                lead_investigator: "security-team".to_string(),
                team: vec!["security-analyst".to_string()],
                started_at: SystemTime::now(),
                ended_at: None,
                findings: Vec::new(),
                evidence: Vec::new(),
                remediation_actions: Vec::new(),
            };

            self.investigations.insert(investigation.id.clone(), investigation);
        }
    }

    /// Start investigation
    pub fn start_investigation(&mut self, breach_id: &Uuid) -> Option<String> {
        if let Some(_breach) = self.breaches.iter().find(|b| b.id == *breach_id) {
            let investigation = BreachInvestigation {
                id: format!("inv-{}", Uuid::new_v4()),
                breach_id: *breach_id,
                status: InvestigationStatus::InProgress,
                lead_investigator: "security-team".to_string(),
                team: Vec::new(),
                started_at: SystemTime::now(),
                ended_at: None,
                findings: Vec::new(),
                evidence: Vec::new(),
                remediation_actions: Vec::new(),
            };

            let investigation_id = investigation.id.clone();
            self.investigations.insert(investigation_id.clone(), investigation);
            Some(investigation_id)
        } else {
            None
        }
    }

    /// Add evidence to investigation
    pub fn add_evidence(&mut self, investigation_id: &str, evidence: Evidence) -> bool {
        if let Some(investigation) = self.investigations.get_mut(investigation_id) {
            investigation.evidence.push(evidence);
            true
        } else {
            false
        }
    }

    /// Add finding to investigation
    pub fn add_finding(&mut self, investigation_id: &str, finding: InvestigationFinding) -> bool {
        if let Some(investigation) = self.investigations.get_mut(investigation_id) {
            investigation.findings.push(finding);
            true
        } else {
            false
        }
    }

    /// Complete investigation
    pub fn complete_investigation(&mut self, investigation_id: &str) -> bool {
        if let Some(investigation) = self.investigations.get_mut(investigation_id) {
            investigation.status = InvestigationStatus::Completed;
            investigation.ended_at = Some(SystemTime::now());
            true
        } else {
            false
        }
    }

    /// Update breach status
    pub fn update_breach_status(&mut self, breach_id: &Uuid, status: BreachStatus) -> bool {
        if let Some(breach) = self.breaches.iter_mut().find(|b| b.id == *breach_id) {
            breach.status = status;
            true
        } else {
            false
        }
    }

    /// Get breaches by status
    pub fn get_breaches_by_status(&self, status: BreachStatus) -> Vec<&DataBreach> {
        self.breaches
            .iter()
            .filter(|breach| breach.status == status)
            .collect()
    }

    /// Get breaches by severity
    pub fn get_breaches_by_severity(&self, severity: BreachSeverity) -> Vec<&DataBreach> {
        self.breaches
            .iter()
            .filter(|breach| breach.severity == severity)
            .collect()
    }

    /// Get overdue notifications
    pub fn get_overdue_notifications(&self) -> Vec<&DataBreach> {
        let now = SystemTime::now();
        self.breaches
            .iter()
            .filter(|breach| {
                breach.notification_required &&
                breach.notification_deadline.map_or(false, |deadline| deadline <= now)
            })
            .collect()
    }

    /// Get active investigations
    pub fn get_active_investigations(&self) -> Vec<&BreachInvestigation> {
        self.investigations
            .values()
            .filter(|inv| matches!(inv.status, InvestigationStatus::InProgress | InvestigationStatus::Initiated))
            .collect()
    }

    /// Get breach statistics
    pub fn get_breach_statistics(&self) -> BreachStatistics {
        let total_breaches = self.breaches.len();
        let critical_breaches = self.get_breaches_by_severity(BreachSeverity::Critical).len();
        let active_investigations = self.get_active_investigations().len();
        let overdue_notifications = self.get_overdue_notifications().len();
        let resolved_breaches = self.get_breaches_by_status(BreachStatus::Resolved).len();

        let resolution_rate = if total_breaches > 0 {
            resolved_breaches as f64 / total_breaches as f64
        } else {
            0.0
        };

        BreachStatistics {
            total_breaches,
            critical_breaches,
            active_investigations,
            overdue_notifications,
            resolved_breaches,
            resolution_rate,
        }
    }

    /// Clean up old breaches
    pub fn cleanup_old_breaches(&mut self, cutoff_time: SystemTime) {
        self.breaches.retain(|breach| breach.discovered_at > cutoff_time);
    }

    /// Calculate risk score for breach
    pub fn calculate_risk_score(&self, breach: &DataBreach) -> f64 {
        let severity_score = match breach.severity {
            BreachSeverity::Low => 0.2,
            BreachSeverity::Medium => 0.4,
            BreachSeverity::High => 0.6,
            BreachSeverity::Critical => 0.8,
            BreachSeverity::Catastrophic => 1.0,
        };

        let volume_score = match breach.affected_data.data_subjects_count {
            0..=100 => 0.1,
            101..=1000 => 0.3,
            1001..=10000 => 0.5,
            10001..=100000 => 0.7,
            _ => 0.9,
        };

        let sensitive_data_score = if breach.affected_data.sensitive_types.is_empty() {
            0.0
        } else {
            0.3
        };

        (severity_score * 0.5) + (volume_score * 0.3) + (sensitive_data_score * 0.2)
    }
}

/// Breach statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachStatistics {
    /// Total breaches
    pub total_breaches: usize,
    /// Critical breaches
    pub critical_breaches: usize,
    /// Active investigations
    pub active_investigations: usize,
    /// Overdue notifications
    pub overdue_notifications: usize,
    /// Resolved breaches
    pub resolved_breaches: usize,
    /// Resolution rate
    pub resolution_rate: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breach_detector_creation() {
        let detector = BreachDetector::new();
        assert_eq!(detector.detection_rules.len(), 0);
        assert_eq!(detector.breaches.len(), 0);
        assert_eq!(detector.investigations.len(), 0);
        assert!(detector.config.real_time_detection);
    }

    #[test]
    fn test_breach_severity_ordering() {
        assert!(BreachSeverity::Catastrophic > BreachSeverity::Critical);
        assert!(BreachSeverity::Critical > BreachSeverity::High);
        assert!(BreachSeverity::High > BreachSeverity::Medium);
        assert!(BreachSeverity::Medium > BreachSeverity::Low);
    }

    #[test]
    fn test_breach_reporting() {
        let mut detector = BreachDetector::new();

        let breach = DataBreach {
            id: Uuid::new_v4(),
            breach_type: BreachType::UnauthorizedAccess,
            severity: BreachSeverity::High,
            discovered_at: SystemTime::now(),
            occurred_at: None,
            affected_data: AffectedData {
                data_subjects_count: 1500,
                data_categories: HashSet::new(),
                sensitive_types: [SensitiveDataType::Pii].into(),
                geographic_scope: ["US".to_string()].into(),
                volume_estimate: DataVolumeEstimate {
                    records: Some(1500),
                    files: None,
                    size_bytes: None,
                    confidence: 0.9,
                },
            },
            source: BreachSource::External,
            status: BreachStatus::Detected,
            notification_required: false,
            notification_deadline: None,
            impact_assessment: ImpactAssessment {
                business_impact: BusinessImpact {
                    service_disruption: false,
                    operational_impact: ImpactLevel::Medium,
                    competitive_loss: false,
                    continuity_impact: ImpactLevel::Low,
                },
                individual_impact: IndividualImpact {
                    privacy_impact: ImpactLevel::High,
                    identity_theft_risk: RiskLevel::Medium,
                    financial_harm_risk: RiskLevel::Low,
                    emotional_distress: ImpactLevel::Medium,
                },
                regulatory_impact: RegulatoryImpact {
                    applicable_regulations: [RegulatoryFramework::Gdpr].into(),
                    violation_severity: ComplianceSeverity::High,
                    potential_fines: Some(50000.0),
                    investigation_likelihood: RiskLevel::High,
                },
                reputational_impact: ReputationalImpact {
                    media_attention: RiskLevel::Medium,
                    customer_trust: ImpactLevel::Medium,
                    brand_impact: ImpactLevel::Medium,
                    social_media_impact: ImpactLevel::Low,
                },
                financial_impact: FinancialImpact {
                    immediate_costs: Some(10000.0),
                    long_term_costs: Some(50000.0),
                    revenue_impact: None,
                    recovery_costs: Some(5000.0),
                },
            },
        };

        detector.report_breach(breach.clone());
        assert_eq!(detector.breaches.len(), 1);

        let reported_breach = &detector.breaches[0];
        assert!(reported_breach.notification_required); // Should be true due to high severity and PII
        assert!(reported_breach.notification_deadline.is_some());
    }

    #[test]
    fn test_investigation_workflow() {
        let mut detector = BreachDetector::new();

        let breach = DataBreach {
            id: Uuid::new_v4(),
            breach_type: BreachType::DataExfiltration,
            severity: BreachSeverity::Critical,
            discovered_at: SystemTime::now(),
            occurred_at: None,
            affected_data: AffectedData {
                data_subjects_count: 100,
                data_categories: HashSet::new(),
                sensitive_types: HashSet::new(),
                geographic_scope: HashSet::new(),
                volume_estimate: DataVolumeEstimate {
                    records: Some(100),
                    files: None,
                    size_bytes: None,
                    confidence: 0.8,
                },
            },
            source: BreachSource::Internal,
            status: BreachStatus::Investigation,
            notification_required: false,
            notification_deadline: None,
            impact_assessment: ImpactAssessment {
                business_impact: BusinessImpact {
                    service_disruption: false,
                    operational_impact: ImpactLevel::Low,
                    competitive_loss: false,
                    continuity_impact: ImpactLevel::Low,
                },
                individual_impact: IndividualImpact {
                    privacy_impact: ImpactLevel::Medium,
                    identity_theft_risk: RiskLevel::Low,
                    financial_harm_risk: RiskLevel::Low,
                    emotional_distress: ImpactLevel::Low,
                },
                regulatory_impact: RegulatoryImpact {
                    applicable_regulations: HashSet::new(),
                    violation_severity: ComplianceSeverity::Medium,
                    potential_fines: None,
                    investigation_likelihood: RiskLevel::Medium,
                },
                reputational_impact: ReputationalImpact {
                    media_attention: RiskLevel::Low,
                    customer_trust: ImpactLevel::Low,
                    brand_impact: ImpactLevel::Low,
                    social_media_impact: ImpactLevel::Low,
                },
                financial_impact: FinancialImpact {
                    immediate_costs: None,
                    long_term_costs: None,
                    revenue_impact: None,
                    recovery_costs: None,
                },
            },
        };

        let breach_id = breach.id;
        detector.report_breach(breach);

        // Start investigation
        let investigation_id = detector.start_investigation(&breach_id);
        assert!(investigation_id.is_some());
        assert_eq!(detector.investigations.len(), 1);

        // Add evidence
        let evidence = Evidence {
            id: "evidence-1".to_string(),
            evidence_type: EvidenceType::SystemLogs,
            description: "Server access logs".to_string(),
            collected_at: SystemTime::now(),
            collector: "forensic-analyst".to_string(),
            chain_of_custody: Vec::new(),
            location: "/var/log/access.log".to_string(),
            checksum: Some("sha256:abc123".to_string()),
        };

        let success = detector.add_evidence(&investigation_id.unwrap(), evidence);
        assert!(success);

        // Complete investigation
        let success = detector.complete_investigation(&investigation_id.unwrap());
        assert!(success);

        let investigation = detector.investigations.get(&investigation_id.unwrap()).unwrap();
        assert_eq!(investigation.status, InvestigationStatus::Completed);
        assert!(investigation.ended_at.is_some());
    }

    #[test]
    fn test_risk_score_calculation() {
        let detector = BreachDetector::new();

        let high_risk_breach = DataBreach {
            id: Uuid::new_v4(),
            breach_type: BreachType::DataExfiltration,
            severity: BreachSeverity::Critical,
            discovered_at: SystemTime::now(),
            occurred_at: None,
            affected_data: AffectedData {
                data_subjects_count: 50000,
                data_categories: HashSet::new(),
                sensitive_types: [SensitiveDataType::Pii, SensitiveDataType::Financial].into(),
                geographic_scope: HashSet::new(),
                volume_estimate: DataVolumeEstimate {
                    records: Some(50000),
                    files: None,
                    size_bytes: None,
                    confidence: 0.9,
                },
            },
            source: BreachSource::External,
            status: BreachStatus::Detected,
            notification_required: true,
            notification_deadline: None,
            impact_assessment: ImpactAssessment {
                business_impact: BusinessImpact {
                    service_disruption: false,
                    operational_impact: ImpactLevel::Low,
                    competitive_loss: false,
                    continuity_impact: ImpactLevel::Low,
                },
                individual_impact: IndividualImpact {
                    privacy_impact: ImpactLevel::Medium,
                    identity_theft_risk: RiskLevel::Low,
                    financial_harm_risk: RiskLevel::Low,
                    emotional_distress: ImpactLevel::Low,
                },
                regulatory_impact: RegulatoryImpact {
                    applicable_regulations: HashSet::new(),
                    violation_severity: ComplianceSeverity::Medium,
                    potential_fines: None,
                    investigation_likelihood: RiskLevel::Medium,
                },
                reputational_impact: ReputationalImpact {
                    media_attention: RiskLevel::Low,
                    customer_trust: ImpactLevel::Low,
                    brand_impact: ImpactLevel::Low,
                    social_media_impact: ImpactLevel::Low,
                },
                financial_impact: FinancialImpact {
                    immediate_costs: None,
                    long_term_costs: None,
                    revenue_impact: None,
                    recovery_costs: None,
                },
            },
        };

        let risk_score = detector.calculate_risk_score(&high_risk_breach);
        assert!(risk_score > 0.7); // Should be high risk
    }

    #[test]
    fn test_impact_and_risk_level_ordering() {
        assert!(ImpactLevel::Severe > ImpactLevel::High);
        assert!(ImpactLevel::High > ImpactLevel::Medium);
        assert!(RiskLevel::VeryHigh > RiskLevel::High);
        assert!(RiskLevel::High > RiskLevel::Medium);
    }
}