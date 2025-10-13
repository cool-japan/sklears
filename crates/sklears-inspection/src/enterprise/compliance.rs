//! Compliance reporting and regulatory requirement management
//!
//! This module provides comprehensive compliance reporting capabilities for explanation systems,
//! supporting various regulatory frameworks and providing automated compliance assessment.

use crate::{SklResult, SklearsError};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Regulatory frameworks that require compliance
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceFramework {
    // European regulations
    /// GDPR

    /// GDPR
    GDPR, // General Data Protection Regulation
    /// AIAct

    /// AIAct
    AIAct, // EU AI Act
    /// MedicalDevice

    /// MedicalDevice
    MedicalDevice, // Medical Device Regulation (MDR)

    // US regulations
    /// HIPAA

    /// HIPAA
    HIPAA, // Health Insurance Portability and Accountability Act
    /// SOX

    /// SOX
    SOX, // Sarbanes-Oxley Act
    /// FCRA

    /// FCRA
    FCRA, // Fair Credit Reporting Act
    /// CCPA

    /// CCPA
    CCPA, // California Consumer Privacy Act

    // Financial regulations
    /// Basel3
    Basel3, // Basel III banking regulations
    /// MiFID2
    MiFID2, // Markets in Financial Instruments Directive
    /// DoddFrank
    DoddFrank, // Dodd-Frank Act
    /// PciDss
    PciDss, // Payment Card Industry Data Security Standard

    // Industry standards
    /// ISO27001
    ISO27001, // Information Security Management
    /// SOC2
    SOC2, // Service Organization Control 2
    /// NIST
    NIST, // NIST Cybersecurity Framework

    // Custom frameworks
    /// Custom
    Custom(String),
}

impl ComplianceFramework {
    /// Get the display name for the framework
    pub fn display_name(&self) -> &str {
        match self {
            ComplianceFramework::GDPR => "General Data Protection Regulation",
            ComplianceFramework::AIAct => "EU AI Act",
            ComplianceFramework::MedicalDevice => "Medical Device Regulation",
            ComplianceFramework::HIPAA => "Health Insurance Portability and Accountability Act",
            ComplianceFramework::SOX => "Sarbanes-Oxley Act",
            ComplianceFramework::FCRA => "Fair Credit Reporting Act",
            ComplianceFramework::CCPA => "California Consumer Privacy Act",
            ComplianceFramework::Basel3 => "Basel III Banking Regulations",
            ComplianceFramework::MiFID2 => "Markets in Financial Instruments Directive",
            ComplianceFramework::DoddFrank => "Dodd-Frank Act",
            ComplianceFramework::PciDss => "Payment Card Industry Data Security Standard",
            ComplianceFramework::ISO27001 => "ISO 27001 Information Security Management",
            ComplianceFramework::SOC2 => "SOC 2 Service Organization Control",
            ComplianceFramework::NIST => "NIST Cybersecurity Framework",
            ComplianceFramework::Custom(name) => name,
        }
    }

    /// Get the regulatory authority for this framework
    pub fn authority(&self) -> &str {
        match self {
            ComplianceFramework::GDPR
            | ComplianceFramework::AIAct
            | ComplianceFramework::MedicalDevice => "European Union",
            ComplianceFramework::HIPAA
            | ComplianceFramework::SOX
            | ComplianceFramework::FCRA
            | ComplianceFramework::DoddFrank => "United States",
            ComplianceFramework::CCPA => "California, USA",
            ComplianceFramework::Basel3 => "Basel Committee on Banking Supervision",
            ComplianceFramework::MiFID2 => "European Securities and Markets Authority",
            ComplianceFramework::PciDss => "PCI Security Standards Council",
            ComplianceFramework::ISO27001 => "International Organization for Standardization",
            ComplianceFramework::SOC2 => "American Institute of CPAs",
            ComplianceFramework::NIST => "National Institute of Standards and Technology",
            ComplianceFramework::Custom(_) => "Custom",
        }
    }

    /// Check if this framework requires explanation transparency
    pub fn requires_explanation_transparency(&self) -> bool {
        matches!(
            self,
            ComplianceFramework::GDPR
                | ComplianceFramework::AIAct
                | ComplianceFramework::FCRA
                | ComplianceFramework::CCPA
        )
    }

    /// Check if this framework has specific AI/ML requirements
    pub fn has_ai_requirements(&self) -> bool {
        matches!(
            self,
            ComplianceFramework::AIAct
                | ComplianceFramework::FCRA
                | ComplianceFramework::MedicalDevice
        )
    }
}

impl std::fmt::Display for ComplianceFramework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplianceFramework::Custom(name) => write!(f, "Custom({})", name),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Risk levels for compliance violations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low

    /// Low
    Low,
    /// Medium

    /// Medium
    Medium,
    /// High

    /// High
    High,
    /// Critical

    /// Critical
    Critical,
}

impl RiskLevel {
    /// Get numeric score for the risk level
    pub fn score(&self) -> u8 {
        match self {
            RiskLevel::Low => 1,
            RiskLevel::Medium => 2,
            RiskLevel::High => 3,
            RiskLevel::Critical => 4,
        }
    }

    /// Check if this risk level requires immediate action
    pub fn requires_immediate_action(&self) -> bool {
        matches!(self, RiskLevel::Critical | RiskLevel::High)
    }
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "low"),
            RiskLevel::Medium => write!(f, "medium"),
            RiskLevel::High => write!(f, "high"),
            RiskLevel::Critical => write!(f, "critical"),
        }
    }
}

/// Compliance status for a requirement or rule
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// Compliant

    /// Compliant
    Compliant,
    /// NonCompliant

    /// NonCompliant
    NonCompliant,
    /// PartiallyCompliant

    /// PartiallyCompliant
    PartiallyCompliant,
    /// NotApplicable

    /// NotApplicable
    NotApplicable,
    /// Unknown

    /// Unknown
    Unknown,
}

impl ComplianceStatus {
    /// Check if this status indicates compliance
    pub fn is_compliant(&self) -> bool {
        matches!(
            self,
            ComplianceStatus::Compliant | ComplianceStatus::NotApplicable
        )
    }

    /// Get risk level associated with this status
    pub fn risk_level(&self) -> RiskLevel {
        match self {
            ComplianceStatus::Compliant | ComplianceStatus::NotApplicable => RiskLevel::Low,
            ComplianceStatus::PartiallyCompliant => RiskLevel::Medium,
            ComplianceStatus::NonCompliant => RiskLevel::High,
            ComplianceStatus::Unknown => RiskLevel::Medium,
        }
    }
}

impl std::fmt::Display for ComplianceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplianceStatus::Compliant => write!(f, "compliant"),
            ComplianceStatus::NonCompliant => write!(f, "non-compliant"),
            ComplianceStatus::PartiallyCompliant => write!(f, "partially compliant"),
            ComplianceStatus::NotApplicable => write!(f, "not applicable"),
            ComplianceStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Individual compliance requirement within a framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    /// Unique rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Associated compliance framework
    pub framework: ComplianceFramework,
    /// Risk level if not compliant
    pub risk_level: RiskLevel,
    /// Whether this rule is mandatory
    pub mandatory: bool,
    /// Rule category (e.g., "data protection", "model transparency")
    pub category: String,
    /// Current compliance status
    pub status: ComplianceStatus,
    /// Assessment criteria
    pub criteria: Vec<String>,
    /// Evidence required for compliance
    pub evidence_required: Vec<String>,
    /// Last assessment date
    pub last_assessed: Option<DateTime<Utc>>,
    /// Next assessment due date
    pub next_assessment_due: Option<DateTime<Utc>>,
    /// Assessment notes
    pub notes: Vec<String>,
    /// Remediation actions if non-compliant
    pub remediation_actions: Vec<String>,
}

impl ComplianceRule {
    /// Create a new compliance rule
    pub fn new(
        name: String,
        description: String,
        framework: ComplianceFramework,
        category: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            description,
            framework,
            risk_level: RiskLevel::Medium,
            mandatory: true,
            category,
            status: ComplianceStatus::Unknown,
            criteria: Vec::new(),
            evidence_required: Vec::new(),
            last_assessed: None,
            next_assessment_due: None,
            notes: Vec::new(),
            remediation_actions: Vec::new(),
        }
    }

    /// Set risk level
    pub fn with_risk_level(mut self, risk_level: RiskLevel) -> Self {
        self.risk_level = risk_level;
        self
    }

    /// Set mandatory status
    pub fn with_mandatory(mut self, mandatory: bool) -> Self {
        self.mandatory = mandatory;
        self
    }

    /// Add assessment criteria
    pub fn with_criteria(mut self, criteria: Vec<String>) -> Self {
        self.criteria = criteria;
        self
    }

    /// Add required evidence
    pub fn with_evidence_required(mut self, evidence: Vec<String>) -> Self {
        self.evidence_required = evidence;
        self
    }

    /// Update compliance status
    pub fn update_status(&mut self, status: ComplianceStatus) {
        self.status = status;
        self.last_assessed = Some(Utc::now());

        // Set next assessment date based on risk level
        let assessment_interval = match self.risk_level {
            RiskLevel::Critical => Duration::days(30),
            RiskLevel::High => Duration::days(90),
            RiskLevel::Medium => Duration::days(180),
            RiskLevel::Low => Duration::days(365),
        };

        self.next_assessment_due = Some(Utc::now() + assessment_interval);
    }

    /// Add assessment note
    pub fn add_note(&mut self, note: String) {
        self.notes.push(note);
    }

    /// Add remediation action
    pub fn add_remediation_action(&mut self, action: String) {
        self.remediation_actions.push(action);
    }

    /// Check if assessment is due
    pub fn is_assessment_due(&self) -> bool {
        if let Some(due_date) = self.next_assessment_due {
            Utc::now() >= due_date
        } else {
            true // No assessment date means it's due
        }
    }
}

/// Regulatory requirement (higher level than individual rules)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryRequirement {
    /// Unique requirement identifier
    pub id: String,
    /// Requirement title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Associated framework
    pub framework: ComplianceFramework,
    /// Legal reference (article, section, etc.)
    pub legal_reference: String,
    /// Implementation deadline
    pub deadline: Option<DateTime<Utc>>,
    /// Priority level
    pub priority: RiskLevel,
    /// Associated compliance rules
    pub rules: Vec<String>,
    /// Implementation status
    pub implementation_status: String,
    /// Owner/responsible party
    pub owner: Option<String>,
    /// Cost of non-compliance
    pub penalty_description: Option<String>,
}

impl RegulatoryRequirement {
    /// Create a new regulatory requirement
    pub fn new(
        title: String,
        description: String,
        framework: ComplianceFramework,
        legal_reference: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title,
            description,
            framework,
            legal_reference,
            deadline: None,
            priority: RiskLevel::Medium,
            rules: Vec::new(),
            implementation_status: "not started".to_string(),
            owner: None,
            penalty_description: None,
        }
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: RiskLevel) -> Self {
        self.priority = priority;
        self
    }

    /// Set owner
    pub fn with_owner(mut self, owner: String) -> Self {
        self.owner = Some(owner);
        self
    }

    /// Add associated rule
    pub fn add_rule(&mut self, rule_id: String) {
        if !self.rules.contains(&rule_id) {
            self.rules.push(rule_id);
        }
    }

    /// Check if implementation is overdue
    pub fn is_overdue(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Utc::now() > deadline
        } else {
            false
        }
    }
}

/// Compliance report for a specific time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report ID
    pub id: String,
    /// Report title
    pub title: String,
    /// Reporting period start
    pub period_start: DateTime<Utc>,
    /// Reporting period end
    pub period_end: DateTime<Utc>,
    /// Report generation time
    pub generated_at: DateTime<Utc>,
    /// Report author
    pub author: Option<String>,
    /// Frameworks covered in this report
    pub frameworks: Vec<ComplianceFramework>,
    /// Overall compliance summary
    pub summary: ComplianceSummary,
    /// Compliance status by framework
    pub framework_compliance: HashMap<ComplianceFramework, FrameworkComplianceStatus>,
    /// Key findings
    pub key_findings: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Risk assessment
    pub risk_assessment: Vec<ComplianceRisk>,
    /// Action items
    pub action_items: Vec<ActionItem>,
}

/// Summary of compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceSummary {
    /// Total number of rules assessed
    pub total_rules: usize,
    /// Number of compliant rules
    pub compliant_rules: usize,
    /// Number of non-compliant rules
    pub non_compliant_rules: usize,
    /// Number of partially compliant rules
    pub partially_compliant_rules: usize,
    /// Overall compliance percentage
    pub compliance_percentage: f64,
    /// Overall risk score
    pub risk_score: f64,
}

impl ComplianceSummary {
    /// Calculate compliance summary from rules
    pub fn from_rules(rules: &[ComplianceRule]) -> Self {
        let total_rules = rules.len();
        let compliant_rules = rules
            .iter()
            .filter(|r| r.status == ComplianceStatus::Compliant)
            .count();
        let non_compliant_rules = rules
            .iter()
            .filter(|r| r.status == ComplianceStatus::NonCompliant)
            .count();
        let partially_compliant_rules = rules
            .iter()
            .filter(|r| r.status == ComplianceStatus::PartiallyCompliant)
            .count();

        let compliance_percentage = if total_rules > 0 {
            (compliant_rules as f64 / total_rules as f64) * 100.0
        } else {
            0.0
        };

        let risk_score = rules
            .iter()
            .map(|rule| {
                let status_multiplier = match rule.status {
                    ComplianceStatus::Compliant | ComplianceStatus::NotApplicable => 0.0,
                    ComplianceStatus::PartiallyCompliant => 0.5,
                    ComplianceStatus::NonCompliant => 1.0,
                    ComplianceStatus::Unknown => 0.3,
                };
                rule.risk_level.score() as f64 * status_multiplier
            })
            .sum::<f64>()
            / total_rules.max(1) as f64;

        Self {
            total_rules,
            compliant_rules,
            non_compliant_rules,
            partially_compliant_rules,
            compliance_percentage,
            risk_score,
        }
    }
}

/// Compliance status for a specific framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkComplianceStatus {
    /// Framework
    pub framework: ComplianceFramework,
    /// Compliance status
    pub status: ComplianceStatus,
    /// Compliance percentage
    pub compliance_percentage: f64,
    /// Number of compliant rules
    pub compliant_rules: usize,
    /// Total number of rules
    pub total_rules: usize,
    /// Key issues
    pub key_issues: Vec<String>,
    /// Next assessment due
    pub next_assessment_due: Option<DateTime<Utc>>,
}

/// Identified compliance risk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRisk {
    /// Risk identifier
    pub id: String,
    /// Risk description
    pub description: String,
    /// Associated framework
    pub framework: ComplianceFramework,
    /// Risk level
    pub level: RiskLevel,
    /// Likelihood of occurrence
    pub likelihood: String,
    /// Potential impact
    pub impact: String,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
    /// Owner
    pub owner: Option<String>,
    /// Due date for mitigation
    pub due_date: Option<DateTime<Utc>>,
}

/// Action item for compliance improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    /// Action item ID
    pub id: String,
    /// Description
    pub description: String,
    /// Priority
    pub priority: RiskLevel,
    /// Assigned to
    pub assigned_to: Option<String>,
    /// Due date
    pub due_date: Option<DateTime<Utc>>,
    /// Status
    pub status: String,
    /// Associated rule or requirement
    pub related_rule: Option<String>,
}

/// Main compliance reporter system
#[derive(Debug)]
pub struct ComplianceReporter {
    /// All compliance rules
    rules: HashMap<String, ComplianceRule>,
    /// All regulatory requirements
    requirements: HashMap<String, RegulatoryRequirement>,
    /// Generated reports
    reports: Vec<ComplianceReport>,
    /// Active frameworks
    active_frameworks: HashSet<ComplianceFramework>,
}

impl ComplianceReporter {
    /// Create a new compliance reporter
    pub fn new() -> Self {
        let mut reporter = Self {
            rules: HashMap::new(),
            requirements: HashMap::new(),
            reports: Vec::new(),
            active_frameworks: HashSet::new(),
        };

        // Add default compliance rules
        reporter.add_default_rules();
        reporter
    }

    /// Add a compliance rule
    pub fn add_rule(&mut self, rule: ComplianceRule) {
        self.active_frameworks.insert(rule.framework.clone());
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Add a regulatory requirement
    pub fn add_requirement(&mut self, requirement: RegulatoryRequirement) {
        self.active_frameworks.insert(requirement.framework.clone());
        self.requirements
            .insert(requirement.id.clone(), requirement);
    }

    /// Update rule status
    pub fn update_rule_status(&mut self, rule_id: &str, status: ComplianceStatus) -> SklResult<()> {
        if let Some(rule) = self.rules.get_mut(rule_id) {
            rule.update_status(status);
            Ok(())
        } else {
            Err(SklearsError::InvalidParameter {
                name: "rule_id".to_string(),
                reason: format!("Rule '{}' not found", rule_id),
            })
        }
    }

    /// Get rules that need assessment
    pub fn get_rules_due_for_assessment(&self) -> Vec<&ComplianceRule> {
        self.rules
            .values()
            .filter(|rule| rule.is_assessment_due())
            .collect()
    }

    /// Get non-compliant rules
    pub fn get_non_compliant_rules(&self) -> Vec<&ComplianceRule> {
        self.rules
            .values()
            .filter(|rule| rule.status == ComplianceStatus::NonCompliant)
            .collect()
    }

    /// Get rules by framework
    pub fn get_rules_by_framework(&self, framework: &ComplianceFramework) -> Vec<&ComplianceRule> {
        self.rules
            .values()
            .filter(|rule| &rule.framework == framework)
            .collect()
    }

    /// Generate compliance report
    pub fn generate_report(
        &mut self,
        title: String,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        author: Option<String>,
        frameworks: Option<Vec<ComplianceFramework>>,
    ) -> SklResult<ComplianceReport> {
        let report_frameworks =
            frameworks.unwrap_or_else(|| self.active_frameworks.iter().cloned().collect());

        // Filter rules for the specified frameworks
        let relevant_rules: Vec<_> = self
            .rules
            .values()
            .filter(|rule| report_frameworks.contains(&rule.framework))
            .cloned()
            .collect();

        // Calculate overall summary
        let summary = ComplianceSummary::from_rules(&relevant_rules);

        // Calculate framework-specific compliance
        let mut framework_compliance = HashMap::new();
        for framework in &report_frameworks {
            let framework_rules: Vec<_> = relevant_rules
                .iter()
                .filter(|rule| &rule.framework == framework)
                .cloned()
                .collect();

            let framework_summary = ComplianceSummary::from_rules(&framework_rules);

            let status = if framework_summary.compliance_percentage >= 95.0 {
                ComplianceStatus::Compliant
            } else if framework_summary.compliance_percentage >= 70.0 {
                ComplianceStatus::PartiallyCompliant
            } else {
                ComplianceStatus::NonCompliant
            };

            let key_issues: Vec<String> = framework_rules
                .iter()
                .filter(|rule| rule.status == ComplianceStatus::NonCompliant)
                .map(|rule| format!("{}: {}", rule.name, rule.description))
                .collect();

            let next_assessment_due = framework_rules
                .iter()
                .filter_map(|rule| rule.next_assessment_due)
                .min();

            framework_compliance.insert(
                framework.clone(),
                /// FrameworkComplianceStatus
                FrameworkComplianceStatus {
                    framework: framework.clone(),
                    status,
                    compliance_percentage: framework_summary.compliance_percentage,
                    compliant_rules: framework_summary.compliant_rules,
                    total_rules: framework_summary.total_rules,
                    key_issues,
                    next_assessment_due,
                },
            );
        }

        // Generate key findings
        let relevant_rule_refs: Vec<&ComplianceRule> = relevant_rules.iter().collect();
        let key_findings = self.generate_key_findings(&relevant_rule_refs, &summary);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&relevant_rule_refs);

        // Generate risk assessment
        let risk_assessment = self.generate_risk_assessment(&relevant_rule_refs);

        // Generate action items
        let action_items = self.generate_action_items(&relevant_rule_refs);

        let report = ComplianceReport {
            id: Uuid::new_v4().to_string(),
            title,
            period_start,
            period_end,
            generated_at: Utc::now(),
            author,
            frameworks: report_frameworks,
            summary,
            framework_compliance,
            key_findings,
            recommendations,
            risk_assessment,
            action_items,
        };

        self.reports.push(report.clone());
        Ok(report)
    }

    /// Get compliance status for a specific framework
    pub fn get_framework_compliance_status(
        &self,
        framework: &ComplianceFramework,
    ) -> ComplianceStatus {
        let rules: Vec<ComplianceRule> = self
            .get_rules_by_framework(framework)
            .into_iter()
            .cloned()
            .collect();
        if rules.is_empty() {
            return ComplianceStatus::NotApplicable;
        }

        let summary = ComplianceSummary::from_rules(&rules);

        if summary.compliance_percentage >= 95.0 {
            ComplianceStatus::Compliant
        } else if summary.compliance_percentage >= 70.0 {
            ComplianceStatus::PartiallyCompliant
        } else {
            ComplianceStatus::NonCompliant
        }
    }

    /// Get overall compliance score
    pub fn get_overall_compliance_score(&self) -> f64 {
        let all_rules: Vec<ComplianceRule> = self.rules.values().cloned().collect();
        let summary = ComplianceSummary::from_rules(&all_rules);
        summary.compliance_percentage
    }

    fn add_default_rules(&mut self) {
        // GDPR rules
        let gdpr_rules = vec![
            ComplianceRule::new(
                "Data Processing Transparency".to_string(),
                "Individuals must be informed about automated decision-making".to_string(),
                ComplianceFramework::GDPR,
                "transparency".to_string(),
            )
            .with_risk_level(RiskLevel::High)
            .with_criteria(vec![
                "Provide clear information about automated processing".to_string(),
                "Explain the logic involved in decision-making".to_string(),
                "Inform about the significance and consequences".to_string(),
            ]),
            ComplianceRule::new(
                "Right to Explanation".to_string(),
                "Individuals have the right to obtain explanations for automated decisions"
                    .to_string(),
                ComplianceFramework::GDPR,
                "individual_rights".to_string(),
            )
            .with_risk_level(RiskLevel::Critical)
            .with_criteria(vec![
                "Provide meaningful explanations upon request".to_string(),
                "Ensure explanations are understandable to data subjects".to_string(),
            ]),
        ];

        for rule in gdpr_rules {
            self.add_rule(rule);
        }

        // AI Act rules
        let ai_act_rules = vec![
            ComplianceRule::new(
                "High-Risk AI System Requirements".to_string(),
                "High-risk AI systems must be transparent and explainable".to_string(),
                ComplianceFramework::AIAct,
                "transparency".to_string(),
            )
            .with_risk_level(RiskLevel::Critical)
            .with_criteria(vec![
                "Ensure appropriate transparency for users".to_string(),
                "Enable users to interpret system output".to_string(),
                "Enable users to use the system appropriately".to_string(),
            ]),
            ComplianceRule::new(
                "Risk Management System".to_string(),
                "Implement and maintain a risk management system".to_string(),
                ComplianceFramework::AIAct,
                "risk_management".to_string(),
            )
            .with_risk_level(RiskLevel::High),
        ];

        for rule in ai_act_rules {
            self.add_rule(rule);
        }

        // FCRA rules
        let fcra_rules = vec![ComplianceRule::new(
            "Adverse Action Notices".to_string(),
            "Provide adverse action notices for automated decisions".to_string(),
            ComplianceFramework::FCRA,
            "disclosure".to_string(),
        )
        .with_risk_level(RiskLevel::High)
        .with_criteria(vec![
            "Send adverse action notice within required timeframe".to_string(),
            "Include specific reasons for adverse action".to_string(),
            "Provide information about consumer rights".to_string(),
        ])];

        for rule in fcra_rules {
            self.add_rule(rule);
        }
    }

    fn generate_key_findings(
        &self,
        rules: &[&ComplianceRule],
        summary: &ComplianceSummary,
    ) -> Vec<String> {
        let mut findings = Vec::new();

        // Overall compliance finding
        if summary.compliance_percentage >= 95.0 {
            findings.push("Overall compliance status is excellent".to_string());
        } else if summary.compliance_percentage >= 80.0 {
            findings.push(
                "Overall compliance status is good with some areas for improvement".to_string(),
            );
        } else {
            findings.push("Overall compliance status requires significant improvement".to_string());
        }

        // High-risk non-compliant rules
        let high_risk_non_compliant: Vec<_> = rules
            .iter()
            .filter(|rule| {
                rule.status == ComplianceStatus::NonCompliant
                    && matches!(rule.risk_level, RiskLevel::High | RiskLevel::Critical)
            })
            .collect();

        if !high_risk_non_compliant.is_empty() {
            findings.push(format!(
                "{} high-risk compliance rules are non-compliant and require immediate attention",
                high_risk_non_compliant.len()
            ));
        }

        // Framework-specific findings
        let frameworks: HashSet<_> = rules.iter().map(|rule| &rule.framework).collect();
        for framework in frameworks {
            let framework_rules: Vec<ComplianceRule> = rules
                .iter()
                .filter(|rule| &rule.framework == framework)
                .map(|&rule| rule.clone())
                .collect();

            let framework_summary = ComplianceSummary::from_rules(&framework_rules);

            if framework_summary.compliance_percentage < 70.0 {
                findings.push(format!(
                    "{} compliance is below acceptable levels ({:.1}%)",
                    framework.display_name(),
                    framework_summary.compliance_percentage
                ));
            }
        }

        findings
    }

    fn generate_recommendations(&self, rules: &[&ComplianceRule]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Recommendations for non-compliant rules
        let non_compliant_rules: Vec<_> = rules
            .iter()
            .filter(|rule| rule.status == ComplianceStatus::NonCompliant)
            .collect();

        if !non_compliant_rules.is_empty() {
            recommendations.push(
                "Prioritize addressing non-compliant rules, starting with highest risk items"
                    .to_string(),
            );

            for rule in non_compliant_rules.iter().take(5) {
                if !rule.remediation_actions.is_empty() {
                    recommendations.push(format!(
                        "For '{}': {}",
                        rule.name,
                        rule.remediation_actions.join("; ")
                    ));
                }
            }
        }

        // Assessment due recommendations
        let assessment_due: Vec<_> = rules
            .iter()
            .filter(|rule| rule.is_assessment_due())
            .collect();

        if !assessment_due.is_empty() {
            recommendations.push(format!(
                "Conduct assessments for {} rules that are due for review",
                assessment_due.len()
            ));
        }

        // General recommendations
        recommendations
            .push("Implement automated compliance monitoring to catch issues early".to_string());
        recommendations.push("Provide regular compliance training to relevant staff".to_string());
        recommendations
            .push("Establish clear processes for handling compliance violations".to_string());

        recommendations
    }

    fn generate_risk_assessment(&self, rules: &[&ComplianceRule]) -> Vec<ComplianceRisk> {
        let mut risks = Vec::new();

        // Generate risks for non-compliant critical rules
        for rule in rules {
            if rule.status == ComplianceStatus::NonCompliant
                && rule.risk_level == RiskLevel::Critical
            {
                risks.push(ComplianceRisk {
                    id: Uuid::new_v4().to_string(),
                    description: format!("Non-compliance with critical rule: {}", rule.name),
                    framework: rule.framework.clone(),
                    level: RiskLevel::Critical,
                    likelihood: "High".to_string(),
                    impact: "Regulatory penalties, legal action, reputation damage".to_string(),
                    mitigation_strategies: rule.remediation_actions.clone(),
                    owner: None,
                    due_date: Some(Utc::now() + Duration::days(30)),
                });
            }
        }

        // Add general compliance risks
        if rules
            .iter()
            .any(|rule| rule.status == ComplianceStatus::Unknown)
        {
            risks.push(ComplianceRisk {
                id: Uuid::new_v4().to_string(),
                description: "Multiple compliance rules have unknown status".to_string(),
                framework: ComplianceFramework::Custom("General".to_string()),
                level: RiskLevel::Medium,
                likelihood: "Medium".to_string(),
                impact: "Potential compliance violations going undetected".to_string(),
                mitigation_strategies: vec![
                    "Conduct comprehensive compliance assessment".to_string(),
                    "Implement automated monitoring".to_string(),
                ],
                owner: None,
                due_date: Some(Utc::now() + Duration::days(60)),
            });
        }

        risks
    }

    fn generate_action_items(&self, rules: &[&ComplianceRule]) -> Vec<ActionItem> {
        let mut actions = Vec::new();

        // Action items for non-compliant rules
        for rule in rules {
            if rule.status == ComplianceStatus::NonCompliant {
                let priority = rule.risk_level.clone();
                let due_date = match priority {
                    RiskLevel::Critical => Some(Utc::now() + Duration::days(7)),
                    RiskLevel::High => Some(Utc::now() + Duration::days(30)),
                    RiskLevel::Medium => Some(Utc::now() + Duration::days(90)),
                    RiskLevel::Low => Some(Utc::now() + Duration::days(180)),
                };

                actions.push(ActionItem {
                    id: Uuid::new_v4().to_string(),
                    description: format!("Address non-compliance for rule: {}", rule.name),
                    priority,
                    assigned_to: None,
                    due_date,
                    status: "open".to_string(),
                    related_rule: Some(rule.id.clone()),
                });
            }
        }

        // Action items for assessment due
        let assessment_due: Vec<_> = rules
            .iter()
            .filter(|rule| rule.is_assessment_due())
            .collect();

        if !assessment_due.is_empty() {
            actions.push(ActionItem {
                id: Uuid::new_v4().to_string(),
                description: format!(
                    "Conduct compliance assessment for {} rules",
                    assessment_due.len()
                ),
                priority: RiskLevel::Medium,
                assigned_to: None,
                due_date: Some(Utc::now() + Duration::days(14)),
                status: "open".to_string(),
                related_rule: None,
            });
        }

        actions
    }
}

impl Default for ComplianceReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_framework() {
        assert_eq!(
            ComplianceFramework::GDPR.display_name(),
            "General Data Protection Regulation"
        );
        assert!(ComplianceFramework::GDPR.requires_explanation_transparency());
        assert!(!ComplianceFramework::GDPR.has_ai_requirements());
        assert!(ComplianceFramework::AIAct.has_ai_requirements());
    }

    #[test]
    fn test_risk_level() {
        assert_eq!(RiskLevel::Critical.score(), 4);
        assert_eq!(RiskLevel::Low.score(), 1);
        assert!(RiskLevel::Critical.requires_immediate_action());
        assert!(!RiskLevel::Low.requires_immediate_action());
    }

    #[test]
    fn test_compliance_status() {
        assert!(ComplianceStatus::Compliant.is_compliant());
        assert!(!ComplianceStatus::NonCompliant.is_compliant());
        assert_eq!(ComplianceStatus::NonCompliant.risk_level(), RiskLevel::High);
    }

    #[test]
    fn test_compliance_rule() {
        let mut rule = ComplianceRule::new(
            "Test Rule".to_string(),
            "Test description".to_string(),
            ComplianceFramework::GDPR,
            "test".to_string(),
        );

        assert_eq!(rule.status, ComplianceStatus::Unknown);
        assert!(rule.is_assessment_due());

        rule.update_status(ComplianceStatus::Compliant);
        assert_eq!(rule.status, ComplianceStatus::Compliant);
        assert!(rule.last_assessed.is_some());
        assert!(rule.next_assessment_due.is_some());
    }

    #[test]
    fn test_compliance_summary() {
        let rules = vec![
            ComplianceRule::new(
                "Rule 1".to_string(),
                "Description 1".to_string(),
                ComplianceFramework::GDPR,
                "test".to_string(),
            )
            .tap_mut(|r| r.status = ComplianceStatus::Compliant),
            ComplianceRule::new(
                "Rule 2".to_string(),
                "Description 2".to_string(),
                ComplianceFramework::GDPR,
                "test".to_string(),
            )
            .tap_mut(|r| r.status = ComplianceStatus::NonCompliant),
        ];

        let summary = ComplianceSummary::from_rules(&rules);
        assert_eq!(summary.total_rules, 2);
        assert_eq!(summary.compliant_rules, 1);
        assert_eq!(summary.non_compliant_rules, 1);
        assert_eq!(summary.compliance_percentage, 50.0);
    }

    #[test]
    fn test_compliance_reporter() {
        let mut reporter = ComplianceReporter::new();

        // Should have some default rules
        assert!(!reporter.rules.is_empty());

        // Test updating rule status
        let rule_ids: Vec<_> = reporter.rules.keys().cloned().collect();
        if let Some(rule_id) = rule_ids.first() {
            reporter
                .update_rule_status(rule_id, ComplianceStatus::Compliant)
                .unwrap();
            assert_eq!(reporter.rules[rule_id].status, ComplianceStatus::Compliant);
        }

        // Test generating report
        let report = reporter
            .generate_report(
                "Test Report".to_string(),
                Utc::now() - Duration::days(30),
                Utc::now(),
                Some("Test Author".to_string()),
                None,
            )
            .unwrap();

        assert_eq!(report.title, "Test Report");
        assert!(!report.frameworks.is_empty());
        assert!(report.summary.total_rules > 0);
    }

    #[test]
    fn test_regulatory_requirement() {
        let mut requirement = RegulatoryRequirement::new(
            "GDPR Article 22".to_string(),
            "Automated individual decision-making".to_string(),
            ComplianceFramework::GDPR,
            "Article 22".to_string(),
        );

        requirement.add_rule("rule1".to_string());
        assert!(requirement.rules.contains(&"rule1".to_string()));

        let deadline = Utc::now() - Duration::days(1);
        requirement.deadline = Some(deadline);
        assert!(requirement.is_overdue());
    }
}

// Helper trait for test modifications
#[cfg(test)]
trait TapMut {
    fn tap_mut<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut Self),
        Self: Sized,
    {
        f(&mut self);
        self
    }
}

#[cfg(test)]
impl<T> TapMut for T {}
