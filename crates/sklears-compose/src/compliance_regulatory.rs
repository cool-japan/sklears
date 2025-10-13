//! Regulatory frameworks, standards, and compliance rule management
//!
//! This module provides comprehensive regulatory framework management including
//! framework implementations, control assessments, remediation planning,
//! gap analysis, and maturity assessment for compliance programs.

use std::{
    collections::{HashMap, HashSet},
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::compliance_core::RegulatoryFramework;

/// Regulatory framework manager
#[derive(Debug)]
pub struct RegulatoryFrameworkManager {
    /// Active frameworks
    pub frameworks: HashMap<RegulatoryFramework, FrameworkImplementation>,
    /// Compliance rules
    pub compliance_rules: Vec<ComplianceRule>,
    /// Framework assessments
    pub assessments: HashMap<RegulatoryFramework, ComplianceAssessment>,
}

/// Framework implementation
#[derive(Debug, Clone)]
pub struct FrameworkImplementation {
    /// Framework type
    pub framework: RegulatoryFramework,
    /// Implementation status
    pub status: ImplementationStatus,
    /// Control implementations
    pub controls: HashMap<String, ControlImplementation>,
    /// Assessment results
    pub assessment_results: Vec<AssessmentResult>,
    /// Remediation plans
    pub remediation_plans: Vec<RemediationPlan>,
}

/// Implementation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationStatus {
    /// Not started
    NotStarted,
    /// In progress
    InProgress,
    /// Implemented
    Implemented,
    /// Validated
    Validated,
    /// Non-compliant
    NonCompliant,
    /// Under review
    UnderReview,
}

/// Control implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlImplementation {
    /// Control ID
    pub control_id: String,
    /// Control name
    pub name: String,
    /// Control description
    pub description: String,
    /// Implementation status
    pub status: ImplementationStatus,
    /// Evidence
    pub evidence: Vec<String>,
    /// Responsible party
    pub responsible_party: String,
    /// Due date
    pub due_date: Option<SystemTime>,
    /// Last assessment
    pub last_assessment: Option<SystemTime>,
}

/// Assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentResult {
    /// Assessment ID
    pub id: Uuid,
    /// Control ID
    pub control_id: String,
    /// Assessment date
    pub assessment_date: SystemTime,
    /// Assessor
    pub assessor: String,
    /// Result
    pub result: AssessmentOutcome,
    /// Findings
    pub findings: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Remediation required
    pub remediation_required: bool,
}

/// Assessment outcomes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssessmentOutcome {
    /// Compliant
    Compliant,
    /// Partially compliant
    PartiallyCompliant,
    /// Non-compliant
    NonCompliant,
    /// Not applicable
    NotApplicable,
    /// Unable to assess
    UnableToAssess,
}

/// Remediation plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationPlan {
    /// Plan ID
    pub id: Uuid,
    /// Control ID
    pub control_id: String,
    /// Plan name
    pub name: String,
    /// Plan description
    pub description: String,
    /// Remediation actions
    pub actions: Vec<RemediationAction>,
    /// Priority
    pub priority: RemediationPriority,
    /// Due date
    pub due_date: SystemTime,
    /// Status
    pub status: RemediationStatus,
}

/// Remediation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationAction {
    /// Action ID
    pub id: String,
    /// Action description
    pub description: String,
    /// Responsible party
    pub responsible_party: String,
    /// Due date
    pub due_date: SystemTime,
    /// Status
    pub status: ActionStatus,
    /// Completion date
    pub completed_date: Option<SystemTime>,
}

/// Remediation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RemediationPriority {
    /// Low priority
    Low = 1,
    /// Medium priority
    Medium = 2,
    /// High priority
    High = 3,
    /// Critical priority
    Critical = 4,
}

/// Remediation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemediationStatus {
    /// Not started
    NotStarted,
    /// In progress
    InProgress,
    /// Completed
    Completed,
    /// Cancelled
    Cancelled,
    /// On hold
    OnHold,
}

/// Action status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionStatus {
    /// Pending
    Pending,
    /// In progress
    InProgress,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Compliance rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Applicable frameworks
    pub frameworks: HashSet<RegulatoryFramework>,
    /// Rule condition
    pub condition: String,
    /// Rule severity
    pub severity: ComplianceSeverity,
    /// Automated check
    pub automated: bool,
    /// Check frequency
    pub frequency: Duration,
}

/// Compliance severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComplianceSeverity {
    /// Informational
    Info = 1,
    /// Low severity
    Low = 2,
    /// Medium severity
    Medium = 3,
    /// High severity
    High = 4,
    /// Critical severity
    Critical = 5,
}

/// Compliance assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceAssessment {
    /// Assessment ID
    pub id: Uuid,
    /// Framework
    pub framework: RegulatoryFramework,
    /// Assessment date
    pub assessment_date: SystemTime,
    /// Overall score
    pub overall_score: f64,
    /// Control results
    pub control_results: HashMap<String, AssessmentResult>,
    /// Gap analysis
    pub gap_analysis: GapAnalysis,
    /// Maturity assessment
    pub maturity_assessment: MaturityAssessment,
}

/// Gap analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysis {
    /// Identified gaps
    pub gaps: Vec<ComplianceGap>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Prioritization
    pub prioritization: Vec<GapPriority>,
}

/// Compliance gap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceGap {
    /// Gap ID
    pub id: String,
    /// Gap description
    pub description: String,
    /// Control ID
    pub control_id: String,
    /// Gap severity
    pub severity: ComplianceSeverity,
    /// Impact assessment
    pub impact: String,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk score
    pub overall_risk: f64,
    /// Risk categories
    pub risk_categories: HashMap<String, f64>,
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Factor name
    pub name: String,
    /// Factor description
    pub description: String,
    /// Risk score
    pub risk_score: f64,
    /// Likelihood
    pub likelihood: f64,
    /// Impact
    pub impact: f64,
}

/// Gap priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapPriority {
    /// Gap ID
    pub gap_id: String,
    /// Priority score
    pub priority_score: f64,
    /// Business impact
    pub business_impact: f64,
    /// Implementation effort
    pub implementation_effort: f64,
    /// Risk reduction
    pub risk_reduction: f64,
}

/// Maturity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaturityAssessment {
    /// Overall maturity level
    pub overall_maturity: MaturityLevel,
    /// Domain maturity levels
    pub domain_maturity: HashMap<String, MaturityLevel>,
    /// Maturity gaps
    pub maturity_gaps: Vec<MaturityGap>,
    /// Improvement roadmap
    pub improvement_roadmap: Vec<MaturityImprovement>,
}

/// Maturity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaturityLevel {
    /// Initial/Ad-hoc
    Initial = 1,
    /// Repeatable
    Repeatable = 2,
    /// Defined
    Defined = 3,
    /// Managed
    Managed = 4,
    /// Optimizing
    Optimizing = 5,
}

impl Display for MaturityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaturityLevel::Initial => write!(f, "Initial"),
            MaturityLevel::Repeatable => write!(f, "Repeatable"),
            MaturityLevel::Defined => write!(f, "Defined"),
            MaturityLevel::Managed => write!(f, "Managed"),
            MaturityLevel::Optimizing => write!(f, "Optimizing"),
        }
    }
}

/// Maturity gap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaturityGap {
    /// Domain name
    pub domain: String,
    /// Current level
    pub current_level: MaturityLevel,
    /// Target level
    pub target_level: MaturityLevel,
    /// Gap description
    pub gap_description: String,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
}

/// Maturity improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaturityImprovement {
    /// Improvement ID
    pub id: String,
    /// Domain
    pub domain: String,
    /// Current level
    pub current_level: MaturityLevel,
    /// Target level
    pub target_level: MaturityLevel,
    /// Improvement actions
    pub actions: Vec<String>,
    /// Timeline
    pub timeline: Duration,
    /// Success metrics
    pub success_metrics: Vec<String>,
}

impl RegulatoryFrameworkManager {
    /// Create a new regulatory framework manager
    pub fn new() -> Self {
        Self {
            frameworks: HashMap::new(),
            compliance_rules: Vec::new(),
            assessments: HashMap::new(),
        }
    }

    /// Add a framework implementation
    pub fn add_framework(&mut self, implementation: FrameworkImplementation) {
        self.frameworks.insert(implementation.framework.clone(), implementation);
    }

    /// Get framework implementation
    pub fn get_framework(&self, framework: &RegulatoryFramework) -> Option<&FrameworkImplementation> {
        self.frameworks.get(framework)
    }

    /// Add compliance rule
    pub fn add_compliance_rule(&mut self, rule: ComplianceRule) {
        self.compliance_rules.push(rule);
    }

    /// Get compliance rules for framework
    pub fn get_rules_for_framework(&self, framework: &RegulatoryFramework) -> Vec<&ComplianceRule> {
        self.compliance_rules
            .iter()
            .filter(|rule| rule.frameworks.contains(framework))
            .collect()
    }

    /// Conduct compliance assessment
    pub fn conduct_assessment(&mut self, framework: RegulatoryFramework) -> ComplianceAssessment {
        let assessment = ComplianceAssessment {
            id: Uuid::new_v4(),
            framework: framework.clone(),
            assessment_date: SystemTime::now(),
            overall_score: 0.75, // Placeholder
            control_results: HashMap::new(),
            gap_analysis: GapAnalysis {
                gaps: Vec::new(),
                risk_assessment: RiskAssessment {
                    overall_risk: 0.3,
                    risk_categories: HashMap::new(),
                    risk_factors: Vec::new(),
                    mitigation_strategies: Vec::new(),
                },
                prioritization: Vec::new(),
            },
            maturity_assessment: MaturityAssessment {
                overall_maturity: MaturityLevel::Defined,
                domain_maturity: HashMap::new(),
                maturity_gaps: Vec::new(),
                improvement_roadmap: Vec::new(),
            },
        };

        self.assessments.insert(framework, assessment.clone());
        assessment
    }

    /// Get latest assessment
    pub fn get_latest_assessment(&self, framework: &RegulatoryFramework) -> Option<&ComplianceAssessment> {
        self.assessments.get(framework)
    }

    /// Update control status
    pub fn update_control_status(&mut self, framework: &RegulatoryFramework, control_id: &str, status: ImplementationStatus) {
        if let Some(implementation) = self.frameworks.get_mut(framework) {
            if let Some(control) = implementation.controls.get_mut(control_id) {
                control.status = status;
            }
        }
    }

    /// Create remediation plan
    pub fn create_remediation_plan(&mut self, framework: &RegulatoryFramework, plan: RemediationPlan) {
        if let Some(implementation) = self.frameworks.get_mut(framework) {
            implementation.remediation_plans.push(plan);
        }
    }

    /// Get frameworks by status
    pub fn get_frameworks_by_status(&self, status: ImplementationStatus) -> Vec<&FrameworkImplementation> {
        self.frameworks
            .values()
            .filter(|impl_| impl_.status == status)
            .collect()
    }

    /// Calculate overall compliance score
    pub fn calculate_compliance_score(&self) -> f64 {
        if self.frameworks.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.frameworks
            .values()
            .map(|impl_| {
                let compliant_controls = impl_.controls
                    .values()
                    .filter(|control| control.status == ImplementationStatus::Implemented || control.status == ImplementationStatus::Validated)
                    .count() as f64;

                if impl_.controls.is_empty() {
                    0.0
                } else {
                    compliant_controls / impl_.controls.len() as f64
                }
            })
            .sum();

        total_score / self.frameworks.len() as f64
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regulatory_framework_manager() {
        let mut manager = RegulatoryFrameworkManager::new();
        assert_eq!(manager.frameworks.len(), 0);

        let implementation = FrameworkImplementation {
            framework: RegulatoryFramework::Gdpr,
            status: ImplementationStatus::InProgress,
            controls: HashMap::new(),
            assessment_results: Vec::new(),
            remediation_plans: Vec::new(),
        };

        manager.add_framework(implementation);
        assert_eq!(manager.frameworks.len(), 1);
        assert!(manager.get_framework(&RegulatoryFramework::Gdpr).is_some());
    }

    #[test]
    fn test_maturity_levels() {
        assert!(MaturityLevel::Optimizing > MaturityLevel::Managed);
        assert!(MaturityLevel::Managed > MaturityLevel::Defined);
        assert_eq!(MaturityLevel::Initial.to_string(), "Initial");
    }

    #[test]
    fn test_compliance_severity() {
        assert!(ComplianceSeverity::Critical > ComplianceSeverity::High);
        assert!(ComplianceSeverity::High > ComplianceSeverity::Medium);
    }

    #[test]
    fn test_remediation_priority() {
        assert!(RemediationPriority::Critical > RemediationPriority::High);
        assert!(RemediationPriority::High > RemediationPriority::Medium);
    }

    #[test]
    fn test_assessment_outcomes() {
        assert_eq!(AssessmentOutcome::Compliant, AssessmentOutcome::Compliant);
        assert_ne!(AssessmentOutcome::Compliant, AssessmentOutcome::NonCompliant);
    }

    #[test]
    fn test_compliance_rule() {
        let mut frameworks = HashSet::new();
        frameworks.insert(RegulatoryFramework::Gdpr);

        let rule = ComplianceRule {
            id: "test-rule".to_string(),
            name: "Test Rule".to_string(),
            frameworks,
            condition: "test condition".to_string(),
            severity: ComplianceSeverity::Medium,
            automated: true,
            frequency: Duration::from_secs(86400), // Daily
        };

        assert_eq!(rule.id, "test-rule");
        assert!(rule.frameworks.contains(&RegulatoryFramework::Gdpr));
    }

    #[test]
    fn test_compliance_score_calculation() {
        let mut manager = RegulatoryFrameworkManager::new();

        // Empty manager should have 0.0 score
        assert_eq!(manager.calculate_compliance_score(), 0.0);

        // Add framework with no controls
        let implementation = FrameworkImplementation {
            framework: RegulatoryFramework::Gdpr,
            status: ImplementationStatus::InProgress,
            controls: HashMap::new(),
            assessment_results: Vec::new(),
            remediation_plans: Vec::new(),
        };
        manager.add_framework(implementation);

        assert_eq!(manager.calculate_compliance_score(), 0.0);
    }
}