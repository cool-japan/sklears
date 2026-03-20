//! Audit trails, reporting, and compliance auditing
//!
//! This module provides comprehensive auditing capabilities including
//! audit trail management, compliance audits, findings tracking,
//! and automated reporting for regulatory compliance.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::compliance_core::RegulatoryFramework;
use crate::compliance_regulatory::{ComplianceSeverity, FindingCategory};

/// Compliance auditor
#[derive(Debug)]
pub struct ComplianceAuditor {
    /// Audit trail
    pub audit_trail: Vec<AuditEntry>,
    /// Active audits
    pub active_audits: HashMap<String, ComplianceAudit>,
    /// Audit reports
    pub audit_reports: VecDeque<AuditReport>,
    /// Configuration
    pub config: AuditorConfig,
}

/// Audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry ID
    pub id: Uuid,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Event type
    pub event_type: AuditEventType,
    /// Actor (user/system)
    pub actor: String,
    /// Resource
    pub resource: String,
    /// Action
    pub action: String,
    /// Result
    pub result: AuditResult,
    /// Details
    pub details: HashMap<String, serde_json::Value>,
    /// Risk score
    pub risk_score: Option<f64>,
}

/// Audit event types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Data access
    DataAccess,
    /// Data modification
    DataModification,
    /// Data deletion
    DataDeletion,
    /// Configuration change
    ConfigurationChange,
    /// Policy violation
    PolicyViolation,
    /// Consent action
    ConsentAction,
    /// Privacy request
    PrivacyRequest,
    /// Security event
    SecurityEvent,
    /// Custom event
    Custom(String),
}

/// Audit result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditResult {
    /// Action successful
    Success,
    /// Action failed
    Failure,
    /// Action denied
    Denied,
    /// Action suspicious
    Suspicious,
    /// Action under investigation
    Investigation,
}

/// Compliance audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceAudit {
    /// Audit ID
    pub id: String,
    /// Audit name
    pub name: String,
    /// Audit type
    pub audit_type: AuditType,
    /// Target frameworks
    pub frameworks: HashSet<RegulatoryFramework>,
    /// Audit scope
    pub scope: AuditScope,
    /// Audit status
    pub status: AuditStatus,
    /// Start date
    pub start_date: SystemTime,
    /// End date
    pub end_date: Option<SystemTime>,
    /// Auditors
    pub auditors: Vec<String>,
    /// Findings
    pub findings: Vec<AuditFinding>,
}

/// Audit types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditType {
    /// Internal audit
    Internal,
    /// External audit
    External,
    /// Regulatory audit
    Regulatory,
    /// Certification audit
    Certification,
    /// Continuous audit
    Continuous,
    /// Custom audit type
    Custom(String),
}

/// Audit scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditScope {
    /// Systems in scope
    pub systems: Vec<String>,
    /// Data assets in scope
    pub data_assets: Vec<String>,
    /// Processes in scope
    pub processes: Vec<String>,
    /// Controls in scope
    pub controls: Vec<String>,
    /// Time period
    pub time_period: (SystemTime, SystemTime),
}

/// Audit status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditStatus {
    /// Planning phase
    Planning,
    /// In progress
    InProgress,
    /// Under review
    UnderReview,
    /// Completed
    Completed,
    /// Cancelled
    Cancelled,
    /// On hold
    OnHold,
}

/// Audit finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditFinding {
    /// Finding ID
    pub id: Uuid,
    /// Finding title
    pub title: String,
    /// Finding description
    pub description: String,
    /// Finding category
    pub category: FindingCategory,
    /// Severity
    pub severity: ComplianceSeverity,
    /// Evidence
    pub evidence: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Management response
    pub management_response: Option<String>,
    /// Due date for remediation
    pub remediation_due_date: Option<SystemTime>,
}

/// Finding categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FindingCategory {
    /// Control deficiency
    ControlDeficiency,
    /// Policy violation
    PolicyViolation,
    /// Process gap
    ProcessGap,
    /// Documentation issue
    DocumentationIssue,
    /// Technical issue
    TechnicalIssue,
    /// Training issue
    TrainingIssue,
    /// Custom category
    Custom(String),
}

/// Audit report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Report ID
    pub id: Uuid,
    /// Audit ID
    pub audit_id: String,
    /// Report date
    pub report_date: SystemTime,
    /// Executive summary
    pub executive_summary: String,
    /// Findings summary
    pub findings_summary: FindingsSummary,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Management comments
    pub management_comments: Option<String>,
    /// Report attachments
    pub attachments: Vec<String>,
}

/// Findings summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingsSummary {
    /// Total findings
    pub total_findings: usize,
    /// Findings by severity
    pub by_severity: HashMap<ComplianceSeverity, usize>,
    /// Findings by category
    pub by_category: HashMap<FindingCategory, usize>,
    /// Findings by framework
    pub by_framework: HashMap<RegulatoryFramework, usize>,
}

/// Auditor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditorConfig {
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Audit trail retention
    pub audit_retention: Duration,
    /// Risk threshold for alerts
    pub risk_threshold: f64,
    /// Automated reporting
    pub automated_reporting: bool,
}

impl Default for AuditorConfig {
    fn default() -> Self {
        Self {
            real_time_monitoring: true,
            audit_retention: Duration::from_secs(7 * 365 * 24 * 60 * 60), // 7 years
            risk_threshold: 0.7,
            automated_reporting: true,
        }
    }
}

impl ComplianceAuditor {
    /// Create a new compliance auditor
    pub fn new() -> Self {
        Self {
            audit_trail: Vec::new(),
            active_audits: HashMap::new(),
            audit_reports: VecDeque::new(),
            config: AuditorConfig::default(),
        }
    }

    /// Log an audit entry
    pub fn log_entry(&mut self, entry: AuditEntry) {
        // Check if entry exceeds risk threshold
        if let Some(risk_score) = entry.risk_score {
            if risk_score > self.config.risk_threshold {
                // Handle high-risk entry (could trigger alerts)
                self.handle_high_risk_entry(&entry);
            }
        }

        self.audit_trail.push(entry);
    }

    /// Start a new audit
    pub fn start_audit(&mut self, audit: ComplianceAudit) {
        self.active_audits.insert(audit.id.clone(), audit);
    }

    /// Complete an audit
    pub fn complete_audit(&mut self, audit_id: &str) -> Option<ComplianceAudit> {
        if let Some(mut audit) = self.active_audits.remove(audit_id) {
            audit.status = AuditStatus::Completed;
            audit.end_date = Some(SystemTime::now());

            // Generate audit report
            let report = self.generate_audit_report(&audit);
            self.audit_reports.push_back(report);

            Some(audit)
        } else {
            None
        }
    }

    /// Add finding to an active audit
    pub fn add_finding(&mut self, audit_id: &str, finding: AuditFinding) {
        if let Some(audit) = self.active_audits.get_mut(audit_id) {
            audit.findings.push(finding);
        }
    }

    /// Get audit trail entries by event type
    pub fn get_entries_by_type(&self, event_type: &AuditEventType) -> Vec<&AuditEntry> {
        self.audit_trail
            .iter()
            .filter(|entry| entry.event_type == *event_type)
            .collect()
    }

    /// Get audit trail entries by actor
    pub fn get_entries_by_actor(&self, actor: &str) -> Vec<&AuditEntry> {
        self.audit_trail
            .iter()
            .filter(|entry| entry.actor == actor)
            .collect()
    }

    /// Get audit trail entries in time range
    pub fn get_entries_in_range(&self, start: SystemTime, end: SystemTime) -> Vec<&AuditEntry> {
        self.audit_trail
            .iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .collect()
    }

    /// Get high-risk entries
    pub fn get_high_risk_entries(&self) -> Vec<&AuditEntry> {
        self.audit_trail
            .iter()
            .filter(|entry| {
                if let Some(risk_score) = entry.risk_score {
                    risk_score > self.config.risk_threshold
                } else {
                    false
                }
            })
            .collect()
    }

    /// Generate audit report
    fn generate_audit_report(&self, audit: &ComplianceAudit) -> AuditReport {
        let findings_summary = self.create_findings_summary(&audit.findings);

        AuditReport {
            id: Uuid::new_v4(),
            audit_id: audit.id.clone(),
            report_date: SystemTime::now(),
            executive_summary: format!("Compliance audit completed for {} with {} findings", audit.name, audit.findings.len()),
            findings_summary,
            recommendations: self.generate_recommendations(&audit.findings),
            management_comments: None,
            attachments: Vec::new(),
        }
    }

    /// Create findings summary
    fn create_findings_summary(&self, findings: &[AuditFinding]) -> FindingsSummary {
        let mut by_severity = HashMap::new();
        let mut by_category = HashMap::new();
        let mut by_framework = HashMap::new();

        for finding in findings {
            *by_severity.entry(finding.severity).or_insert(0) += 1;
            *by_category.entry(finding.category.clone()).or_insert(0) += 1;
            // Framework association would need additional context
        }

        FindingsSummary {
            total_findings: findings.len(),
            by_severity,
            by_category,
            by_framework,
        }
    }

    /// Generate recommendations based on findings
    fn generate_recommendations(&self, findings: &[AuditFinding]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let critical_findings = findings.iter()
            .filter(|f| f.severity == ComplianceSeverity::Critical)
            .count();

        if critical_findings > 0 {
            recommendations.push(format!("Address {} critical findings immediately", critical_findings));
        }

        let control_deficiencies = findings.iter()
            .filter(|f| f.category == FindingCategory::ControlDeficiency)
            .count();

        if control_deficiencies > 0 {
            recommendations.push("Review and strengthen control implementations".to_string());
        }

        recommendations
    }

    /// Handle high-risk audit entry
    fn handle_high_risk_entry(&self, _entry: &AuditEntry) {
        // Placeholder for alert handling
        // In a real implementation, this might send notifications,
        // create incidents, or trigger automated responses
    }

    /// Get audit statistics
    pub fn get_audit_statistics(&self) -> AuditStatistics {
        let total_entries = self.audit_trail.len();
        let high_risk_entries = self.get_high_risk_entries().len();
        let active_audits = self.active_audits.len();
        let completed_audits = self.audit_reports.len();

        let success_rate = if total_entries > 0 {
            let successful_entries = self.audit_trail
                .iter()
                .filter(|entry| entry.result == AuditResult::Success)
                .count();
            successful_entries as f64 / total_entries as f64
        } else {
            0.0
        };

        AuditStatistics {
            total_entries,
            high_risk_entries,
            active_audits,
            completed_audits,
            success_rate,
        }
    }

    /// Clean up old audit trail entries
    pub fn cleanup_old_entries(&mut self) {
        let retention_cutoff = SystemTime::now()
            .checked_sub(self.config.audit_retention)
            .unwrap_or(SystemTime::UNIX_EPOCH);

        self.audit_trail.retain(|entry| entry.timestamp > retention_cutoff);
    }
}

/// Audit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStatistics {
    /// Total audit entries
    pub total_entries: usize,
    /// High-risk entries
    pub high_risk_entries: usize,
    /// Active audits
    pub active_audits: usize,
    /// Completed audits
    pub completed_audits: usize,
    /// Success rate
    pub success_rate: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_auditor_creation() {
        let auditor = ComplianceAuditor::new();
        assert_eq!(auditor.audit_trail.len(), 0);
        assert_eq!(auditor.active_audits.len(), 0);
        assert!(auditor.config.real_time_monitoring);
    }

    #[test]
    fn test_audit_entry_logging() {
        let mut auditor = ComplianceAuditor::new();

        let entry = AuditEntry {
            id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::DataAccess,
            actor: "test-user".to_string(),
            resource: "test-resource".to_string(),
            action: "read".to_string(),
            result: AuditResult::Success,
            details: HashMap::new(),
            risk_score: Some(0.3),
        };

        auditor.log_entry(entry);
        assert_eq!(auditor.audit_trail.len(), 1);
    }

    #[test]
    fn test_audit_lifecycle() {
        let mut auditor = ComplianceAuditor::new();

        let audit = ComplianceAudit {
            id: "test-audit".to_string(),
            name: "Test Audit".to_string(),
            audit_type: AuditType::Internal,
            frameworks: HashSet::new(),
            scope: AuditScope {
                systems: vec!["system1".to_string()],
                data_assets: vec!["asset1".to_string()],
                processes: vec!["process1".to_string()],
                controls: vec!["control1".to_string()],
                time_period: (SystemTime::now(), SystemTime::now()),
            },
            status: AuditStatus::Planning,
            start_date: SystemTime::now(),
            end_date: None,
            auditors: vec!["auditor1".to_string()],
            findings: Vec::new(),
        };

        // Start audit
        auditor.start_audit(audit);
        assert_eq!(auditor.active_audits.len(), 1);

        // Complete audit
        let completed_audit = auditor.complete_audit("test-audit");
        assert!(completed_audit.is_some());
        assert_eq!(auditor.active_audits.len(), 0);
        assert_eq!(auditor.audit_reports.len(), 1);
    }

    #[test]
    fn test_findings_summary() {
        let findings = vec![
            AuditFinding {
                id: Uuid::new_v4(),
                title: "Test Finding 1".to_string(),
                description: "Description".to_string(),
                category: FindingCategory::ControlDeficiency,
                severity: ComplianceSeverity::High,
                evidence: Vec::new(),
                recommendations: Vec::new(),
                management_response: None,
                remediation_due_date: None,
            },
            AuditFinding {
                id: Uuid::new_v4(),
                title: "Test Finding 2".to_string(),
                description: "Description".to_string(),
                category: FindingCategory::PolicyViolation,
                severity: ComplianceSeverity::Critical,
                evidence: Vec::new(),
                recommendations: Vec::new(),
                management_response: None,
                remediation_due_date: None,
            },
        ];

        let auditor = ComplianceAuditor::new();
        let summary = auditor.create_findings_summary(&findings);

        assert_eq!(summary.total_findings, 2);
        assert_eq!(summary.by_severity.get(&ComplianceSeverity::High), Some(&1));
        assert_eq!(summary.by_severity.get(&ComplianceSeverity::Critical), Some(&1));
    }

    #[test]
    fn test_high_risk_entry_detection() {
        let mut auditor = ComplianceAuditor::new();
        auditor.config.risk_threshold = 0.5;

        let high_risk_entry = AuditEntry {
            id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::SecurityEvent,
            actor: "test-user".to_string(),
            resource: "sensitive-data".to_string(),
            action: "unauthorized_access".to_string(),
            result: AuditResult::Denied,
            details: HashMap::new(),
            risk_score: Some(0.8),
        };

        auditor.log_entry(high_risk_entry);
        let high_risk_entries = auditor.get_high_risk_entries();
        assert_eq!(high_risk_entries.len(), 1);
    }
}