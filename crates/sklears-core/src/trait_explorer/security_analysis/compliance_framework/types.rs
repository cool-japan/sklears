//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::type_aliases::{
    ApprovalWorkflow, CertificationProcess, CertificationRequirement, ComplianceDocumentMapping,
    DashboardManagement, DataAggregation, DetailedAssessmentReporting, DistributionManagement,
    DocumentAccessControls, DocumentLifecycle, DocumentType, DocumentVersionControl,
    DocumentationRequirements, ExecutiveSummaryGeneration, MaintenanceRequirements, PreAssessment,
    PreparationActivity, ReadinessAssessment, RecertificationPlanning, RegulatorySubmissions,
    ReportTemplate, RetentionPolicy, StakeholderReporting, TemplateManagement, TrendReporting,
    VisualizationTool,
};
use super::types_5::ControlsAssessmentEntry;
use super::types_6::{
    AuditType, CertificationReadinessAssessment, CertificationStatusResult, CertificationType,
    ComplianceActionPlan, ComplianceError, ComplianceFrameworkType, ComplianceLevel,
    CompliancePenalty, ComplianceRecommendation, ComplianceRiskAssessment, ComplianceRiskItem,
    ComplianceStatus, FrameworkAssessmentResult, GapAnalysisResult, PolicyComplianceResult,
    RegulatoryControl, SecurityStandardsComplianceResult,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryFramework {
    pub framework_id: String,
    pub name: String,
    pub description: String,
    pub jurisdiction: Vec<String>,
    pub framework_type: ComplianceFrameworkType,
    pub version: String,
    pub effective_date: SystemTime,
    pub requirements: Vec<RegulatoryRequirement>,
    pub controls: Vec<RegulatoryControl>,
    pub assessment_procedures: Vec<AssessmentProcedure>,
    pub penalties: Vec<CompliancePenalty>,
    pub exemptions: Vec<ComplianceExemption>,
    pub update_frequency: Duration,
}
impl RegulatoryFramework {
    pub fn new_gdpr() -> Self {
        Self {
            framework_id: "GDPR".to_string(),
            name: "General Data Protection Regulation".to_string(),
            description: "EU regulation on the protection of personal data and privacy".to_string(),
            jurisdiction: vec!["EU".to_string(), "EEA".to_string()],
            framework_type: ComplianceFrameworkType::GDPR,
            version: "2016/679".to_string(),
            effective_date: SystemTime::now(),
            requirements: vec![RegulatoryRequirement {
                requirement_id: "Article 5".to_string(),
                description: "Principles relating to processing of personal data".to_string(),
                mandatory: true,
            }],
            controls: vec![RegulatoryControl {
                control_id: "GDPR-C1".to_string(),
                name: "Data anonymization".to_string(),
                description: "Pseudonymize or anonymize personal data where possible".to_string(),
            }],
            assessment_procedures: Vec::new(),
            penalties: vec![CompliancePenalty {
                violation_type: "Article 83 infringement".to_string(),
                description: "Administrative fines for GDPR violations".to_string(),
                estimated_financial_impact: 20_000_000.0,
            }],
            exemptions: Vec::new(),
            update_frequency: Duration::from_secs(86400 * 365),
        }
    }
    pub fn new_hipaa() -> Self {
        Self {
            framework_id: "HIPAA".to_string(),
            name: "Health Insurance Portability and Accountability Act".to_string(),
            description: "US regulation protecting sensitive patient health information"
                .to_string(),
            jurisdiction: vec!["US".to_string()],
            framework_type: ComplianceFrameworkType::HIPAA,
            version: "1996".to_string(),
            effective_date: SystemTime::now(),
            requirements: vec![RegulatoryRequirement {
                requirement_id: "164.308(a)(1)".to_string(),
                description: "Security management process".to_string(),
                mandatory: true,
            }],
            controls: vec![RegulatoryControl {
                control_id: "HIPAA-C1".to_string(),
                name: "Access control".to_string(),
                description: "Restrict access to electronic protected health information"
                    .to_string(),
            }],
            assessment_procedures: Vec::new(),
            penalties: vec![CompliancePenalty {
                violation_type: "Willful neglect".to_string(),
                description: "Civil monetary penalties for HIPAA violations".to_string(),
                estimated_financial_impact: 1_500_000.0,
            }],
            exemptions: Vec::new(),
            update_frequency: Duration::from_secs(86400 * 365),
        }
    }
    pub fn new_ccpa() -> Self {
        Self {
            framework_id: "CCPA".to_string(),
            name: "California Consumer Privacy Act".to_string(),
            description: "California state law enhancing privacy rights for consumers".to_string(),
            jurisdiction: vec!["California".to_string(), "US".to_string()],
            framework_type: ComplianceFrameworkType::CCPA,
            version: "2018".to_string(),
            effective_date: SystemTime::now(),
            requirements: vec![RegulatoryRequirement {
                requirement_id: "1798.100".to_string(),
                description: "Consumer right to know about data collection".to_string(),
                mandatory: true,
            }],
            controls: vec![RegulatoryControl {
                control_id: "CCPA-C1".to_string(),
                name: "Opt-out mechanism".to_string(),
                description: "Provide a mechanism for consumers to opt out of data sale"
                    .to_string(),
            }],
            assessment_procedures: Vec::new(),
            penalties: vec![CompliancePenalty {
                violation_type: "Unintentional violation".to_string(),
                description: "Civil penalties per violation".to_string(),
                estimated_financial_impact: 2_500.0,
            }],
            exemptions: Vec::new(),
            update_frequency: Duration::from_secs(86400 * 365),
        }
    }
    pub fn new_sox() -> Self {
        Self {
            framework_id: "SOX".to_string(),
            name: "Sarbanes-Oxley Act".to_string(),
            description:
                "US federal law governing financial reporting and corporate accountability"
                    .to_string(),
            jurisdiction: vec!["US".to_string()],
            framework_type: ComplianceFrameworkType::SOX,
            version: "2002".to_string(),
            effective_date: SystemTime::now(),
            requirements: vec![RegulatoryRequirement {
                requirement_id: "Section 404".to_string(),
                description: "Management assessment of internal controls".to_string(),
                mandatory: true,
            }],
            controls: vec![RegulatoryControl {
                control_id: "SOX-C1".to_string(),
                name: "Change control".to_string(),
                description: "Documented approval process for financial system changes".to_string(),
            }],
            assessment_procedures: Vec::new(),
            penalties: vec![CompliancePenalty {
                violation_type: "Certification violation".to_string(),
                description: "Criminal and civil penalties for non-compliance".to_string(),
                estimated_financial_impact: 5_000_000.0,
            }],
            exemptions: Vec::new(),
            update_frequency: Duration::from_secs(86400 * 365),
        }
    }
    pub fn new_ferpa() -> Self {
        Self {
            framework_id: "FERPA".to_string(),
            name: "Family Educational Rights and Privacy Act".to_string(),
            description: "US federal law protecting the privacy of student education records"
                .to_string(),
            jurisdiction: vec!["US".to_string()],
            framework_type: ComplianceFrameworkType::FERPA,
            version: "1974".to_string(),
            effective_date: SystemTime::now(),
            requirements: vec![RegulatoryRequirement {
                requirement_id: "34 CFR 99.31".to_string(),
                description: "Conditions for disclosure of education records".to_string(),
                mandatory: true,
            }],
            controls: vec![RegulatoryControl {
                control_id: "FERPA-C1".to_string(),
                name: "Consent tracking".to_string(),
                description: "Track and record consent for disclosure of education records"
                    .to_string(),
            }],
            assessment_procedures: Vec::new(),
            penalties: vec![CompliancePenalty {
                violation_type: "Improper disclosure".to_string(),
                description: "Loss of federal funding eligibility".to_string(),
                estimated_financial_impact: 0.0,
            }],
            exemptions: Vec::new(),
            update_frequency: Duration::from_secs(86400 * 365),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCoverageAssessment {
    pub subject_id: String,
    pub status: ComplianceStatus,
    pub items_met: u32,
    pub items_total: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryRequirement {
    pub requirement_id: String,
    pub description: String,
    pub mandatory: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSummary {
    pub total_evidence_items: u32,
    pub evidence_types: Vec<String>,
    pub collection_methods: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationManager {
    pub manager_id: String,
    pub document_types: Vec<DocumentType>,
    pub document_lifecycle: DocumentLifecycle,
    pub version_control: DocumentVersionControl,
    pub approval_workflows: Vec<ApprovalWorkflow>,
    pub distribution_management: DistributionManagement,
    pub retention_policies: Vec<RetentionPolicy>,
    pub access_controls: DocumentAccessControls,
    pub template_management: TemplateManagement,
    pub compliance_mapping: ComplianceDocumentMapping,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlsAssessmentResult {
    pub controls_assessments: Vec<ControlsAssessmentEntry>,
    pub overall_controls_effectiveness: f64,
    pub controls_gaps: Vec<String>,
    pub compensating_controls_analysis: Vec<String>,
    pub controls_maturity: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReportingEngine {
    pub engine_id: String,
    pub report_templates: Vec<ReportTemplate>,
    pub data_aggregation: DataAggregation,
    pub visualization_tools: Vec<VisualizationTool>,
    pub dashboard_management: DashboardManagement,
    pub stakeholder_reporting: StakeholderReporting,
    pub regulatory_submissions: RegulatorySubmissions,
    pub executive_summaries: ExecutiveSummaryGeneration,
    pub detailed_assessments: DetailedAssessmentReporting,
    pub trend_reporting: TrendReporting,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    pub audit_id: String,
    pub audit_type: AuditType,
    pub findings_count: u32,
    pub summary: String,
    pub conducted_at: SystemTime,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryComplianceResult {
    pub regulatory_assessments: Vec<ComplianceCoverageAssessment>,
    pub overall_regulatory_status: ComplianceStatus,
    pub jurisdiction_compliance: HashMap<String, ComplianceStatus>,
    pub regulatory_risks: Vec<ComplianceRiskItem>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceExemption {
    pub exemption_id: String,
    pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationManager {
    pub certification_id: String,
    pub certification_type: CertificationType,
    pub certification_requirements: Vec<CertificationRequirement>,
    pub readiness_assessment: ReadinessAssessment,
    pub preparation_activities: Vec<PreparationActivity>,
    pub documentation_requirements: DocumentationRequirements,
    pub pre_assessment: PreAssessment,
    pub certification_process: CertificationProcess,
    pub maintenance_requirements: MaintenanceRequirements,
    pub recertification_planning: RecertificationPlanning,
}
impl CertificationManager {
    pub fn assess_certification_readiness(
        &self,
        context: &TraitUsageContext,
    ) -> Result<CertificationReadinessAssessment, ComplianceError> {
        let mut readiness_score: f64 = 50.0;
        if context.has_audit_logging {
            readiness_score += 15.0;
        }
        if context.has_access_controls {
            readiness_score += 15.0;
        }
        if context.has_encryption {
            readiness_score += 10.0;
        }
        if context.has_data_anonymization {
            readiness_score += 10.0;
        }
        Ok(CertificationReadinessAssessment {
            certification_id: self.certification_id.clone(),
            certification_type: self.certification_type.clone(),
            readiness_score: readiness_score.min(100.0),
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceAssessmentResult {
    pub assessment_id: String,
    pub assessment_timestamp: SystemTime,
    pub framework_assessments: HashMap<String, FrameworkAssessmentResult>,
    pub regulatory_compliance: RegulatoryComplianceResult,
    pub security_standards_compliance: SecurityStandardsComplianceResult,
    pub audit_results: Vec<AuditResult>,
    pub policy_compliance: PolicyComplianceResult,
    pub controls_assessment: ControlsAssessmentResult,
    pub gap_analysis: GapAnalysisResult,
    pub certification_status: CertificationStatusResult,
    pub compliance_score: f64,
    pub compliance_level: ComplianceLevel,
    pub recommendations: Vec<ComplianceRecommendation>,
    pub action_plan: ComplianceActionPlan,
    pub risk_assessment: ComplianceRiskAssessment,
    pub assessment_confidence: f64,
    pub next_assessment_date: SystemTime,
}
