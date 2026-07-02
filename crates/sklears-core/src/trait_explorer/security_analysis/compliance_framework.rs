use super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFrameworkManager {
    compliance_engines: HashMap<String, ComplianceEngine>,
    regulatory_frameworks: HashMap<String, RegulatoryFramework>,
    security_standards: HashMap<String, SecurityStandard>,
    audit_managers: Vec<AuditManager>,
    policy_engines: Vec<PolicyEngine>,
    controls_assessors: Vec<ControlsAssessor>,
    gap_analyzers: Vec<GapAnalyzer>,
    certification_managers: Vec<CertificationManager>,
    compliance_monitors: Vec<ComplianceMonitor>,
    reporting_engines: Vec<ComplianceReportingEngine>,
    documentation_managers: Vec<DocumentationManager>,
    compliance_config: ComplianceConfiguration,
    compliance_cache: HashMap<String, CachedComplianceResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceEngine {
    engine_id: String,
    framework_type: ComplianceFrameworkType,
    compliance_checkers: Vec<ComplianceChecker>,
    evidence_collectors: Vec<EvidenceCollector>,
    assessment_tools: Vec<AssessmentTool>,
    validation_rules: Vec<ValidationRule>,
    compliance_metrics: ComplianceMetrics,
    automated_testing: AutomatedComplianceTesting,
    continuous_monitoring: ContinuousComplianceMonitoring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceFrameworkType {
    NIST,
    GDPR,
    HIPAA,
    SOC2,
    ISO27001,
    PciDss,
    FERPA,
    CCPA,
    SOX,
    FISMA,
    CommonCriteria,
    FIPS140_2,
    Custom(String),
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStandard {
    pub standard_id: String,
    pub name: String,
    pub organization: String,
    pub version: String,
    pub publication_date: SystemTime,
    pub standard_type: SecurityStandardType,
    pub security_objectives: Vec<SecurityObjective>,
    pub security_controls: Vec<SecurityControl>,
    pub implementation_guidance: Vec<ImplementationGuide>,
    pub assessment_criteria: Vec<AssessmentCriterion>,
    pub maturity_levels: Vec<MaturityLevel>,
    pub cross_references: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityStandardType {
    Technical,
    Operational,
    Management,
    Physical,
    Legal,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditManager {
    pub audit_id: String,
    pub audit_type: AuditType,
    pub audit_scope: AuditScope,
    pub audit_procedures: Vec<AuditProcedure>,
    pub evidence_requirements: Vec<EvidenceRequirement>,
    pub audit_trail_manager: AuditTrailManager,
    pub findings_manager: FindingsManager,
    pub remediation_tracker: RemediationTracker,
    pub audit_reporting: AuditReporting,
    pub quality_assurance: AuditQualityAssurance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditType {
    Internal,
    External,
    Regulatory,
    Certification,
    Surveillance,
    Forensic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEngine {
    pub policy_id: String,
    pub policy_framework: PolicyFramework,
    pub policy_documents: Vec<PolicyDocument>,
    pub policy_enforcement: PolicyEnforcement,
    pub policy_monitoring: PolicyMonitoring,
    pub policy_exceptions: Vec<PolicyException>,
    pub policy_lifecycle: PolicyLifecycle,
    pub policy_assessment: PolicyAssessment,
    pub policy_training: PolicyTraining,
    pub policy_communication: PolicyCommunication,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlsAssessor {
    pub assessor_id: String,
    pub assessment_methodology: AssessmentMethodology,
    pub control_families: Vec<ControlFamily>,
    pub assessment_procedures: Vec<ControlAssessmentProcedure>,
    pub testing_methods: Vec<ControlTestingMethod>,
    pub maturity_assessment: MaturityAssessment,
    pub effectiveness_measurement: EffectivenessMeasurement,
    pub risk_assessment_integration: RiskAssessmentIntegration,
    pub compensating_controls: CompensatingControlsAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalyzer {
    pub analyzer_id: String,
    pub gap_analysis_methodology: GapAnalysisMethodology,
    pub current_state_assessment: CurrentStateAssessment,
    pub target_state_definition: TargetStateDefinition,
    pub gap_identification: GapIdentification,
    pub prioritization_framework: PrioritizationFramework,
    pub remediation_planning: RemediationPlanning,
    pub cost_benefit_analysis: CostBenefitAnalysis,
    pub timeline_estimation: TimelineEstimation,
    pub risk_impact_analysis: RiskImpactAnalysis,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificationType {
    ISO27001,
    SOC2,
    PciDss,
    FIPS140_2,
    CommonCriteria,
    HITRUST,
    FedRAMP,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMonitor {
    pub monitor_id: String,
    pub monitoring_scope: MonitoringScope,
    pub monitoring_frequency: MonitoringFrequency,
    pub automated_checks: Vec<AutomatedComplianceCheck>,
    pub manual_assessments: Vec<ManualAssessment>,
    pub real_time_monitoring: RealTimeMonitoring,
    pub alerting_system: ComplianceAlertingSystem,
    pub trend_analysis: ComplianceTrendAnalysis,
    pub deviation_detection: DeviationDetection,
    pub corrective_actions: CorrectiveActionSystem,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkAssessmentResult {
    pub framework_type: ComplianceFrameworkType,
    pub compliance_status: ComplianceStatus,
    pub requirement_assessments: Vec<RequirementAssessment>,
    pub control_assessments: Vec<ControlAssessment>,
    pub evidence_summary: EvidenceSummary,
    pub findings: Vec<ComplianceFinding>,
    pub remediation_items: Vec<RemediationItem>,
    pub compliance_percentage: f64,
    pub maturity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComplianceStatus {
    Compliant,
    PartiallyCompliant,
    NonCompliant,
    NotApplicable,
    NotAssessed,
    InProgress,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceLevel {
    Basic,
    Intermediate,
    Advanced,
    Expert,
    Optimized,
}

// Result-substructure types: shallow, flat-data types (mirroring `FrameworkAssessmentResult`
// above) produced by the input-dependent logic in the `impl` blocks below.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCoverageAssessment {
    pub subject_id: String,
    pub status: ComplianceStatus,
    pub items_met: u32,
    pub items_total: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryComplianceResult {
    pub regulatory_assessments: Vec<ComplianceCoverageAssessment>,
    pub overall_regulatory_status: ComplianceStatus,
    pub jurisdiction_compliance: HashMap<String, ComplianceStatus>,
    pub regulatory_risks: Vec<ComplianceRiskItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStandardsComplianceResult {
    pub standards_assessments: Vec<ComplianceCoverageAssessment>,
    pub overall_standards_status: ComplianceStatus,
    pub cross_reference_analysis: Vec<String>,
    pub standards_gaps: Vec<String>,
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
pub struct PolicyComplianceResult {
    pub policy_assessments: Vec<PolicyComplianceAssessment>,
    pub overall_policy_status: ComplianceStatus,
    pub policy_violations: Vec<String>,
    pub policy_effectiveness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyComplianceAssessment {
    pub policy_id: String,
    pub status: ComplianceStatus,
    pub summary: String,
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
pub struct ControlsAssessmentEntry {
    pub assessor_id: String,
    pub effectiveness_score: f64,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysisResult {
    pub gap_analyses: Vec<GapAnalysisEntry>,
    pub consolidated_gaps: Vec<ComplianceRiskItem>,
    pub prioritized_gaps: Vec<ComplianceRiskItem>,
    pub remediation_roadmap: Vec<String>,
    pub cost_impact_analysis: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysisEntry {
    pub analyzer_id: String,
    pub summary: String,
    pub severity: RiskSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationStatusResult {
    pub certification_assessments: Vec<CertificationReadinessAssessment>,
    pub overall_readiness: f64,
    pub certification_timeline: Duration,
    pub preparation_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationReadinessAssessment {
    pub certification_id: String,
    pub certification_type: CertificationType,
    pub readiness_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRecommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub priority: MitigationPriority,
    pub related_framework: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceActionPlan {
    pub plan_id: String,
    pub actions: Vec<String>,
    pub estimated_completion: SystemTime,
    pub estimated_cost: EstimatedCost,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRiskAssessment {
    pub identified_risks: Vec<ComplianceRiskItem>,
    pub overall_risk_level: RiskLevel,
    pub risk_mitigation_priority: MitigationPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRiskItem {
    pub risk_id: String,
    pub description: String,
    pub severity: RiskSeverity,
    pub related_framework: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequirementAssessment {
    pub requirement_id: String,
    pub category: String,
    pub status: ComplianceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlAssessment {
    pub control_id: String,
    pub control_type: String,
    pub effective: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSummary {
    pub total_evidence_items: u32,
    pub evidence_types: Vec<String>,
    pub collection_methods: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFinding {
    pub finding_id: String,
    pub description: String,
    pub severity: RiskSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationItem {
    pub item_id: String,
    pub description: String,
    pub priority: MitigationPriority,
    pub estimated_effort: ImplementationEffort,
}

/// A single concrete compliance violation discovered during an assessment or audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub violation_id: String,
    pub framework: String,
    pub requirement_id: String,
    pub description: String,
    pub severity: RiskSeverity,
    pub discovered_date: SystemTime,
}

/// A [`ComplianceViolation`] together with supporting evidence and remediation guidance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolationDetail {
    pub violation: ComplianceViolation,
    pub evidence: Vec<String>,
    pub remediation_guidance: String,
    pub remediation_deadline: Option<SystemTime>,
}

// Regulatory framework / security standard leaf types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryRequirement {
    pub requirement_id: String,
    pub description: String,
    pub mandatory: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryControl {
    pub control_id: String,
    pub name: String,
    pub description: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompliancePenalty {
    pub violation_type: String,
    pub description: String,
    pub estimated_financial_impact: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceExemption {
    pub exemption_id: String,
    pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityObjective {
    pub objective_id: String,
    pub description: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityControl {
    pub control_id: String,
    pub name: String,
    pub category: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationGuide {
    pub guide_id: String,
    pub title: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentCriterion {
    pub criterion_id: String,
    pub description: String,
    pub weight: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaturityLevel {
    pub level: u8,
    pub name: String,
}

// Terse placeholder types for the pluggable audit/policy/controls/gap/certification/monitoring/
// reporting/documentation sub-systems below. None of the managers that own these fields are
// populated by `ComplianceFrameworkManager::new()` (they are extension points for pluggable
// custom implementations), so their nested configuration slots are simple string aliases.

pub type AuditScope = String;
pub type AuditProcedure = String;
pub type EvidenceRequirement = String;
pub type AuditTrailManager = String;
pub type FindingsManager = String;
pub type RemediationTracker = String;
pub type AuditReporting = String;
pub type AuditQualityAssurance = String;

pub type PolicyFramework = String;
pub type PolicyDocument = String;
pub type PolicyEnforcement = String;
pub type PolicyMonitoring = String;
pub type PolicyException = String;
pub type PolicyLifecycle = String;
pub type PolicyAssessment = String;
pub type PolicyTraining = String;
pub type PolicyCommunication = String;

pub type AssessmentMethodology = String;
pub type ControlFamily = String;
pub type ControlAssessmentProcedure = String;
pub type ControlTestingMethod = String;
pub type MaturityAssessment = String;
pub type EffectivenessMeasurement = String;
pub type RiskAssessmentIntegration = String;
pub type CompensatingControlsAssessment = String;

pub type GapAnalysisMethodology = String;
pub type CurrentStateAssessment = String;
pub type TargetStateDefinition = String;
pub type GapIdentification = String;
pub type PrioritizationFramework = String;
pub type RemediationPlanning = String;
pub type CostBenefitAnalysis = String;
pub type TimelineEstimation = String;
pub type RiskImpactAnalysis = String;

pub type CertificationRequirement = String;
pub type ReadinessAssessment = String;
pub type PreparationActivity = String;
pub type DocumentationRequirements = String;
pub type PreAssessment = String;
pub type CertificationProcess = String;
pub type MaintenanceRequirements = String;
pub type RecertificationPlanning = String;

pub type MonitoringScope = String;
pub type MonitoringFrequency = String;
pub type AutomatedComplianceCheck = String;
pub type ManualAssessment = String;
pub type RealTimeMonitoring = String;
pub type ComplianceAlertingSystem = String;
pub type ComplianceTrendAnalysis = String;
pub type DeviationDetection = String;
pub type CorrectiveActionSystem = String;

pub type ReportTemplate = String;
pub type DataAggregation = String;
pub type VisualizationTool = String;
pub type DashboardManagement = String;
pub type StakeholderReporting = String;
pub type RegulatorySubmissions = String;
pub type ExecutiveSummaryGeneration = String;
pub type DetailedAssessmentReporting = String;
pub type TrendReporting = String;

pub type DocumentType = String;
pub type DocumentLifecycle = String;
pub type DocumentVersionControl = String;
pub type ApprovalWorkflow = String;
pub type DistributionManagement = String;
pub type RetentionPolicy = String;
pub type DocumentAccessControls = String;
pub type TemplateManagement = String;
pub type ComplianceDocumentMapping = String;

impl ComplianceFrameworkManager {
    pub fn new() -> Self {
        Self {
            compliance_engines: Self::initialize_compliance_engines(),
            regulatory_frameworks: Self::initialize_regulatory_frameworks(),
            security_standards: Self::initialize_security_standards(),
            audit_managers: Vec::new(),
            policy_engines: Vec::new(),
            controls_assessors: Vec::new(),
            gap_analyzers: Vec::new(),
            certification_managers: Vec::new(),
            compliance_monitors: Vec::new(),
            reporting_engines: Vec::new(),
            documentation_managers: Vec::new(),
            compliance_config: ComplianceConfiguration::default(),
            compliance_cache: HashMap::new(),
        }
    }

    pub fn assess_compliance(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<ComplianceAssessmentResult, ComplianceError> {
        let assessment_id = self.generate_assessment_id(context);

        if let Some(cached_result) = self.get_cached_result(&assessment_id) {
            if self.is_cache_valid(&cached_result) {
                return Ok(cached_result.result.clone());
            }
        }

        let framework_assessments = self.assess_frameworks(context)?;
        let regulatory_compliance = self.assess_regulatory_compliance(context)?;
        let security_standards_compliance = self.assess_security_standards_compliance(context)?;
        let audit_results = self.conduct_audits(context)?;
        let policy_compliance = self.assess_policy_compliance(context)?;
        let controls_assessment = self.assess_controls(context)?;
        let gap_analysis = self.perform_gap_analysis(context)?;
        let certification_status = self.assess_certification_status(context)?;

        let compliance_score = self.calculate_compliance_score(
            &framework_assessments,
            &regulatory_compliance,
            &security_standards_compliance,
            &controls_assessment,
        )?;

        let compliance_level = self.determine_compliance_level(compliance_score)?;
        let recommendations = self.generate_compliance_recommendations(
            &framework_assessments,
            &gap_analysis,
            &controls_assessment,
        )?;

        let action_plan = self.develop_compliance_action_plan(&recommendations, &gap_analysis)?;
        let risk_assessment = self.assess_compliance_risks(context, &framework_assessments)?;
        let assessment_confidence = self.calculate_assessment_confidence()?;
        let next_assessment_date = self.calculate_next_assessment_date(&framework_assessments)?;

        let result = ComplianceAssessmentResult {
            assessment_id: assessment_id.clone(),
            assessment_timestamp: SystemTime::now(),
            framework_assessments,
            regulatory_compliance,
            security_standards_compliance,
            audit_results,
            policy_compliance,
            controls_assessment,
            gap_analysis,
            certification_status,
            compliance_score,
            compliance_level,
            recommendations,
            action_plan,
            risk_assessment,
            assessment_confidence,
            next_assessment_date,
        };

        self.cache_result(assessment_id, &result);
        Ok(result)
    }

    fn assess_frameworks(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<HashMap<String, FrameworkAssessmentResult>, ComplianceError> {
        let mut results = HashMap::new();

        for (framework_name, engine) in &self.compliance_engines {
            let assessment_result = engine.assess_framework_compliance(context)?;
            results.insert(framework_name.clone(), assessment_result);
        }

        Ok(results)
    }

    fn assess_regulatory_compliance(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<RegulatoryComplianceResult, ComplianceError> {
        let mut regulatory_assessments = Vec::new();

        for framework in self.regulatory_frameworks.values() {
            let assessment = self.assess_regulatory_framework(framework, context)?;
            regulatory_assessments.push(assessment);
        }

        let overall_regulatory_status =
            self.calculate_overall_regulatory_status(&regulatory_assessments)?;
        let jurisdiction_compliance =
            self.assess_jurisdiction_compliance(&regulatory_assessments)?;
        let regulatory_risks = self.identify_regulatory_risks(&regulatory_assessments)?;

        Ok(RegulatoryComplianceResult {
            regulatory_assessments,
            overall_regulatory_status,
            jurisdiction_compliance,
            regulatory_risks,
        })
    }

    fn assess_security_standards_compliance(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<SecurityStandardsComplianceResult, ComplianceError> {
        let mut standards_assessments = Vec::new();

        for standard in self.security_standards.values() {
            let assessment = self.assess_security_standard(standard, context)?;
            standards_assessments.push(assessment);
        }

        let overall_standards_status =
            self.calculate_overall_standards_status(&standards_assessments)?;
        let cross_reference_analysis = self.analyze_cross_references(&standards_assessments)?;
        let standards_gaps = self.identify_standards_gaps(&standards_assessments)?;

        Ok(SecurityStandardsComplianceResult {
            standards_assessments,
            overall_standards_status,
            cross_reference_analysis,
            standards_gaps,
        })
    }

    fn conduct_audits(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<Vec<AuditResult>, ComplianceError> {
        let mut audit_results = Vec::new();

        for audit_manager in &self.audit_managers {
            let result = audit_manager.conduct_audit(context)?;
            audit_results.push(result);
        }

        Ok(audit_results)
    }

    fn assess_policy_compliance(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<PolicyComplianceResult, ComplianceError> {
        let mut policy_assessments = Vec::new();

        for policy_engine in &self.policy_engines {
            let assessment = policy_engine.assess_policy_compliance(context)?;
            policy_assessments.push(assessment);
        }

        let overall_policy_status = self.calculate_overall_policy_status(&policy_assessments)?;
        let policy_violations = self.identify_policy_violations(&policy_assessments)?;
        let policy_effectiveness = self.assess_policy_effectiveness(&policy_assessments)?;

        Ok(PolicyComplianceResult {
            policy_assessments,
            overall_policy_status,
            policy_violations,
            policy_effectiveness,
        })
    }

    fn assess_controls(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<ControlsAssessmentResult, ComplianceError> {
        let mut controls_assessments = Vec::new();

        for assessor in &self.controls_assessors {
            let assessment = assessor.assess_controls(context)?;
            controls_assessments.push(assessment);
        }

        let overall_controls_effectiveness =
            self.calculate_overall_controls_effectiveness(&controls_assessments)?;
        let controls_gaps = self.identify_controls_gaps(&controls_assessments)?;
        let compensating_controls_analysis =
            self.analyze_compensating_controls(&controls_assessments)?;
        let controls_maturity = self.assess_controls_maturity(&controls_assessments)?;

        Ok(ControlsAssessmentResult {
            controls_assessments,
            overall_controls_effectiveness,
            controls_gaps,
            compensating_controls_analysis,
            controls_maturity,
        })
    }

    fn perform_gap_analysis(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<GapAnalysisResult, ComplianceError> {
        let mut gap_analyses = Vec::new();

        for analyzer in &self.gap_analyzers {
            let analysis = analyzer.perform_gap_analysis(context)?;
            gap_analyses.push(analysis);
        }

        let consolidated_gaps = self.consolidate_gaps(&gap_analyses)?;
        let prioritized_gaps = self.prioritize_gaps(&consolidated_gaps)?;
        let remediation_roadmap = self.develop_remediation_roadmap(&prioritized_gaps)?;
        let cost_impact_analysis = self.analyze_cost_impact(&remediation_roadmap)?;

        Ok(GapAnalysisResult {
            gap_analyses,
            consolidated_gaps,
            prioritized_gaps,
            remediation_roadmap,
            cost_impact_analysis,
        })
    }

    fn assess_certification_status(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<CertificationStatusResult, ComplianceError> {
        let mut certification_assessments = Vec::new();

        for certification_manager in &self.certification_managers {
            let assessment = certification_manager.assess_certification_readiness(context)?;
            certification_assessments.push(assessment);
        }

        let overall_readiness =
            self.calculate_overall_certification_readiness(&certification_assessments)?;
        let certification_timeline =
            self.estimate_certification_timeline(&certification_assessments)?;
        let preparation_requirements =
            self.identify_preparation_requirements(&certification_assessments)?;

        Ok(CertificationStatusResult {
            certification_assessments,
            overall_readiness,
            certification_timeline,
            preparation_requirements,
        })
    }

    fn initialize_compliance_engines() -> HashMap<String, ComplianceEngine> {
        let mut engines = HashMap::new();

        engines.insert("NIST".to_string(), ComplianceEngine::new_nist());
        engines.insert("GDPR".to_string(), ComplianceEngine::new_gdpr());
        engines.insert("HIPAA".to_string(), ComplianceEngine::new_hipaa());
        engines.insert("SOC2".to_string(), ComplianceEngine::new_soc2());
        engines.insert("ISO27001".to_string(), ComplianceEngine::new_iso27001());
        engines.insert("PCI_DSS".to_string(), ComplianceEngine::new_pci_dss());

        engines
    }

    fn initialize_regulatory_frameworks() -> HashMap<String, RegulatoryFramework> {
        let mut frameworks = HashMap::new();

        frameworks.insert("GDPR".to_string(), RegulatoryFramework::new_gdpr());
        frameworks.insert("HIPAA".to_string(), RegulatoryFramework::new_hipaa());
        frameworks.insert("CCPA".to_string(), RegulatoryFramework::new_ccpa());
        frameworks.insert("SOX".to_string(), RegulatoryFramework::new_sox());
        frameworks.insert("FERPA".to_string(), RegulatoryFramework::new_ferpa());

        frameworks
    }

    fn initialize_security_standards() -> HashMap<String, SecurityStandard> {
        let mut standards = HashMap::new();

        standards.insert("ISO27001".to_string(), SecurityStandard::new_iso27001());
        standards.insert("NIST_CSF".to_string(), SecurityStandard::new_nist_csf());
        standards.insert("COBIT".to_string(), SecurityStandard::new_cobit());
        standards.insert("ITIL".to_string(), SecurityStandard::new_itil());
        standards.insert(
            "CIS_Controls".to_string(),
            SecurityStandard::new_cis_controls(),
        );

        standards
    }
}

impl ComplianceFrameworkManager {
    /// Get the configured compliance monitors (pluggable extension point, external access).
    pub fn compliance_monitors(&self) -> &[ComplianceMonitor] {
        &self.compliance_monitors
    }
    /// Get the configured compliance reporting engines (external access).
    pub fn reporting_engines(&self) -> &[ComplianceReportingEngine] {
        &self.reporting_engines
    }
    /// Get the configured documentation managers (external access).
    pub fn documentation_managers(&self) -> &[DocumentationManager] {
        &self.documentation_managers
    }

    /// Derive an overall [`ComplianceStatus`] for `context` by running a full compliance
    /// assessment and reducing the per-framework statuses returned by it to a single value.
    pub fn check_compliance_status(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<ComplianceStatus, ComplianceError> {
        let assessment = self.assess_compliance(context)?;
        Ok(assessment
            .framework_assessments
            .values()
            .map(|a| a.compliance_status.clone())
            .min()
            .unwrap_or(ComplianceStatus::NotAssessed))
    }

    fn generate_assessment_id(&self, context: &TraitUsageContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        context.hash(&mut hasher);
        format!("compliance_assessment_{:x}", hasher.finish())
    }

    fn get_cached_result(&self, assessment_id: &str) -> Option<CachedComplianceResult> {
        self.compliance_cache.get(assessment_id).cloned()
    }

    fn is_cache_valid(&self, cached: &CachedComplianceResult) -> bool {
        if !self.compliance_config.automated_monitoring {
            return false;
        }
        SystemTime::now()
            .duration_since(cached.cache_timestamp)
            .map(|elapsed| elapsed < cached.cache_ttl)
            .unwrap_or(false)
    }

    fn cache_result(&mut self, assessment_id: String, result: &ComplianceAssessmentResult) {
        let cached = CachedComplianceResult {
            result: result.clone(),
            cache_timestamp: SystemTime::now(),
            cache_ttl: Duration::from_secs(3600),
        };
        self.compliance_cache.insert(assessment_id, cached);
    }

    fn calculate_compliance_score(
        &self,
        framework_assessments: &HashMap<String, FrameworkAssessmentResult>,
        regulatory_compliance: &RegulatoryComplianceResult,
        security_standards_compliance: &SecurityStandardsComplianceResult,
        controls_assessment: &ControlsAssessmentResult,
    ) -> Result<f64, ComplianceError> {
        if framework_assessments.is_empty() {
            return Ok(0.0);
        }

        let framework_avg = framework_assessments
            .values()
            .map(|assessment| assessment.compliance_percentage)
            .sum::<f64>()
            / framework_assessments.len() as f64;
        let regulatory_component =
            if regulatory_compliance.overall_regulatory_status == ComplianceStatus::Compliant {
                100.0
            } else {
                60.0
            };
        let standards_component = if security_standards_compliance.overall_standards_status
            == ComplianceStatus::Compliant
        {
            100.0
        } else {
            60.0
        };
        let weighted = framework_avg * 0.4
            + regulatory_component * 0.2
            + standards_component * 0.2
            + controls_assessment.overall_controls_effectiveness * 0.2;
        Ok((weighted / 10.0).clamp(0.0, 10.0))
    }

    fn determine_compliance_level(&self, score: f64) -> Result<ComplianceLevel, ComplianceError> {
        Ok(match score {
            s if s >= 9.0 => ComplianceLevel::Optimized,
            s if s >= 7.5 => ComplianceLevel::Expert,
            s if s >= 6.0 => ComplianceLevel::Advanced,
            s if s >= 4.0 => ComplianceLevel::Intermediate,
            _ => ComplianceLevel::Basic,
        })
    }

    fn generate_compliance_recommendations(
        &self,
        framework_assessments: &HashMap<String, FrameworkAssessmentResult>,
        gap_analysis: &GapAnalysisResult,
        controls_assessment: &ControlsAssessmentResult,
    ) -> Result<Vec<ComplianceRecommendation>, ComplianceError> {
        let mut recommendations = Vec::new();

        for (framework_name, assessment) in framework_assessments {
            if assessment.compliance_status != ComplianceStatus::Compliant {
                let priority = if assessment.compliance_status == ComplianceStatus::NonCompliant {
                    MitigationPriority::Critical
                } else {
                    MitigationPriority::High
                };
                let description = format!(
                    "{framework_name} is currently {:?} ({:.1}% of requirements met); address the outstanding findings to reach full compliance.",
                    assessment.compliance_status, assessment.compliance_percentage
                );
                recommendations.push(ComplianceRecommendation {
                    recommendation_id: format!("REC-{framework_name}"),
                    title: format!("Improve {framework_name} compliance"),
                    description,
                    priority,
                    related_framework: framework_name.clone(),
                });
            }
        }

        for gap in &gap_analysis.prioritized_gaps {
            let priority = match gap.severity {
                RiskSeverity::Critical => MitigationPriority::Critical,
                RiskSeverity::High => MitigationPriority::High,
                RiskSeverity::Medium => MitigationPriority::Medium,
                RiskSeverity::Low => MitigationPriority::Low,
            };
            recommendations.push(ComplianceRecommendation {
                recommendation_id: format!("REC-GAP-{}", gap.risk_id),
                title: "Close identified compliance gap".to_string(),
                description: gap.description.clone(),
                priority,
                related_framework: gap.related_framework.clone(),
            });
        }

        if controls_assessment.overall_controls_effectiveness < 70.0 {
            let description = format!(
                "Overall controls effectiveness is {:.1}%, below the recommended 70% threshold.",
                controls_assessment.overall_controls_effectiveness
            );
            recommendations.push(ComplianceRecommendation {
                recommendation_id: "REC-CONTROLS-001".to_string(),
                title: "Strengthen compliance controls".to_string(),
                description,
                priority: MitigationPriority::High,
                related_framework: "General".to_string(),
            });
        }

        Ok(recommendations)
    }

    fn develop_compliance_action_plan(
        &self,
        recommendations: &[ComplianceRecommendation],
        gap_analysis: &GapAnalysisResult,
    ) -> Result<ComplianceActionPlan, ComplianceError> {
        let actions: Vec<String> = recommendations
            .iter()
            .map(|recommendation| recommendation.title.clone())
            .chain(gap_analysis.remediation_roadmap.iter().cloned())
            .collect();
        let estimated_cost = match actions.len() {
            0..=2 => EstimatedCost::Low,
            3..=6 => EstimatedCost::Medium,
            _ => EstimatedCost::High,
        };
        Ok(ComplianceActionPlan {
            plan_id: format!("PLAN-{}", actions.len()),
            actions,
            estimated_completion: SystemTime::now() + Duration::from_secs(86400 * 90),
            estimated_cost,
        })
    }

    fn assess_compliance_risks(
        &self,
        context: &TraitUsageContext,
        framework_assessments: &HashMap<String, FrameworkAssessmentResult>,
    ) -> Result<ComplianceRiskAssessment, ComplianceError> {
        let mut identified_risks = Vec::new();

        if context.handles_personal_data && !context.has_data_anonymization {
            identified_risks.push(ComplianceRiskItem {
                risk_id: "COMP-RISK-001".to_string(),
                description: "Personal data is processed without anonymization safeguards"
                    .to_string(),
                severity: RiskSeverity::High,
                related_framework: "GDPR".to_string(),
            });
        }
        if context.handles_sensitive_data && !context.has_encryption {
            identified_risks.push(ComplianceRiskItem {
                risk_id: "COMP-RISK-002".to_string(),
                description: "Sensitive data is handled without encryption".to_string(),
                severity: RiskSeverity::Critical,
                related_framework: "HIPAA".to_string(),
            });
        }
        if !context.has_audit_logging {
            identified_risks.push(ComplianceRiskItem {
                risk_id: "COMP-RISK-003".to_string(),
                description: "No audit logging is in place for compliance-relevant operations"
                    .to_string(),
                severity: RiskSeverity::Medium,
                related_framework: "SOC2".to_string(),
            });
        }
        if !context.has_access_controls {
            identified_risks.push(ComplianceRiskItem {
                risk_id: "COMP-RISK-004".to_string(),
                description: "Access controls are not enforced before sensitive operations"
                    .to_string(),
                severity: RiskSeverity::High,
                related_framework: "ISO27001".to_string(),
            });
        }
        for assessment in framework_assessments.values() {
            if assessment.compliance_status == ComplianceStatus::NonCompliant {
                identified_risks.push(ComplianceRiskItem {
                    risk_id: format!("COMP-RISK-{:?}", assessment.framework_type),
                    description: "Framework assessment reported non-compliance".to_string(),
                    severity: RiskSeverity::Critical,
                    related_framework: format!("{:?}", assessment.framework_type),
                });
            }
        }

        let overall_risk_level = identified_risks
            .iter()
            .map(|risk| match risk.severity {
                RiskSeverity::Critical => RiskLevel::Critical,
                RiskSeverity::High => RiskLevel::High,
                RiskSeverity::Medium => RiskLevel::Medium,
                RiskSeverity::Low => RiskLevel::Low,
            })
            .max()
            .unwrap_or(RiskLevel::Minimal);

        let risk_mitigation_priority = if identified_risks
            .iter()
            .any(|risk| risk.severity == RiskSeverity::Critical)
        {
            MitigationPriority::Critical
        } else if identified_risks.is_empty() {
            MitigationPriority::Low
        } else {
            MitigationPriority::High
        };

        Ok(ComplianceRiskAssessment {
            identified_risks,
            overall_risk_level,
            risk_mitigation_priority,
        })
    }

    fn calculate_assessment_confidence(&self) -> Result<f64, ComplianceError> {
        let engine_factor = (self.compliance_engines.len() as f64 / 6.0).min(1.0);
        Ok((0.6 + 0.4 * engine_factor).clamp(0.0, 1.0))
    }

    fn calculate_next_assessment_date(
        &self,
        framework_assessments: &HashMap<String, FrameworkAssessmentResult>,
    ) -> Result<SystemTime, ComplianceError> {
        let has_non_compliant = framework_assessments
            .values()
            .any(|assessment| assessment.compliance_status == ComplianceStatus::NonCompliant);
        let interval = if has_non_compliant {
            Duration::from_secs(86400 * 30)
        } else {
            self.compliance_config
                .assessment_frequency
                .values()
                .min()
                .copied()
                .unwrap_or(Duration::from_secs(86400 * 90))
        };
        Ok(SystemTime::now() + interval)
    }

    fn assess_regulatory_framework(
        &self,
        framework: &RegulatoryFramework,
        context: &TraitUsageContext,
    ) -> Result<ComplianceCoverageAssessment, ComplianceError> {
        let items_total = framework.requirements.len() as u32;
        let mut items_met = items_total;
        if context.handles_personal_data && !context.has_data_anonymization {
            items_met = items_met.saturating_sub(1);
        }
        if !context.has_encryption {
            items_met = items_met.saturating_sub(1);
        }
        if !context.has_audit_logging {
            items_met = items_met.saturating_sub(1);
        }
        let status = compliance_status_from_coverage(items_met, items_total);
        Ok(ComplianceCoverageAssessment {
            subject_id: framework.framework_id.clone(),
            status,
            items_met,
            items_total,
        })
    }

    fn calculate_overall_regulatory_status(
        &self,
        assessments: &[ComplianceCoverageAssessment],
    ) -> Result<ComplianceStatus, ComplianceError> {
        Ok(assessments
            .iter()
            .map(|assessment| assessment.status.clone())
            .min()
            .unwrap_or(ComplianceStatus::NotAssessed))
    }

    fn assess_jurisdiction_compliance(
        &self,
        assessments: &[ComplianceCoverageAssessment],
    ) -> Result<HashMap<String, ComplianceStatus>, ComplianceError> {
        Ok(assessments
            .iter()
            .map(|assessment| (assessment.subject_id.clone(), assessment.status.clone()))
            .collect())
    }

    fn identify_regulatory_risks(
        &self,
        assessments: &[ComplianceCoverageAssessment],
    ) -> Result<Vec<ComplianceRiskItem>, ComplianceError> {
        Ok(assessments
            .iter()
            .filter(|assessment| assessment.status != ComplianceStatus::Compliant)
            .map(|assessment| ComplianceRiskItem {
                risk_id: format!("REG-RISK-{}", assessment.subject_id),
                description: format!(
                    "{} met only {} of {} requirements",
                    assessment.subject_id, assessment.items_met, assessment.items_total
                ),
                severity: if assessment.status == ComplianceStatus::NonCompliant {
                    RiskSeverity::Critical
                } else {
                    RiskSeverity::Medium
                },
                related_framework: assessment.subject_id.clone(),
            })
            .collect())
    }

    fn assess_security_standard(
        &self,
        standard: &SecurityStandard,
        context: &TraitUsageContext,
    ) -> Result<ComplianceCoverageAssessment, ComplianceError> {
        let items_total = standard.security_objectives.len() as u32;
        let mut items_met = items_total;
        if !context.has_access_controls {
            items_met = items_met.saturating_sub(1);
        }
        if context.has_unsafe_operations && !context.has_bounds_checking {
            items_met = items_met.saturating_sub(1);
        }
        let status = compliance_status_from_coverage(items_met, items_total);
        Ok(ComplianceCoverageAssessment {
            subject_id: standard.standard_id.clone(),
            status,
            items_met,
            items_total,
        })
    }

    fn calculate_overall_standards_status(
        &self,
        assessments: &[ComplianceCoverageAssessment],
    ) -> Result<ComplianceStatus, ComplianceError> {
        Ok(assessments
            .iter()
            .map(|assessment| assessment.status.clone())
            .min()
            .unwrap_or(ComplianceStatus::NotAssessed))
    }

    fn analyze_cross_references(
        &self,
        assessments: &[ComplianceCoverageAssessment],
    ) -> Result<Vec<String>, ComplianceError> {
        Ok(assessments
            .iter()
            .map(|assessment| {
                format!(
                    "{} cross-references {} other standards",
                    assessment.subject_id,
                    self.security_standards.len().saturating_sub(1)
                )
            })
            .collect())
    }

    fn identify_standards_gaps(
        &self,
        assessments: &[ComplianceCoverageAssessment],
    ) -> Result<Vec<String>, ComplianceError> {
        Ok(assessments
            .iter()
            .filter(|assessment| assessment.status != ComplianceStatus::Compliant)
            .map(|assessment| {
                format!(
                    "{}: {} of {} objectives outstanding",
                    assessment.subject_id,
                    assessment.items_total.saturating_sub(assessment.items_met),
                    assessment.items_total
                )
            })
            .collect())
    }

    fn calculate_overall_policy_status(
        &self,
        assessments: &[PolicyComplianceAssessment],
    ) -> Result<ComplianceStatus, ComplianceError> {
        Ok(assessments
            .iter()
            .map(|assessment| assessment.status.clone())
            .min()
            .unwrap_or(ComplianceStatus::NotAssessed))
    }

    fn identify_policy_violations(
        &self,
        assessments: &[PolicyComplianceAssessment],
    ) -> Result<Vec<String>, ComplianceError> {
        Ok(assessments
            .iter()
            .filter(|assessment| assessment.status == ComplianceStatus::NonCompliant)
            .map(|assessment| format!("{}: {}", assessment.policy_id, assessment.summary))
            .collect())
    }

    fn assess_policy_effectiveness(
        &self,
        assessments: &[PolicyComplianceAssessment],
    ) -> Result<f64, ComplianceError> {
        if assessments.is_empty() {
            return Ok(100.0);
        }
        let compliant = assessments
            .iter()
            .filter(|assessment| assessment.status == ComplianceStatus::Compliant)
            .count();
        Ok(100.0 * compliant as f64 / assessments.len() as f64)
    }

    fn calculate_overall_controls_effectiveness(
        &self,
        assessments: &[ControlsAssessmentEntry],
    ) -> Result<f64, ComplianceError> {
        if assessments.is_empty() {
            return Ok(100.0);
        }
        Ok(assessments
            .iter()
            .map(|assessment| assessment.effectiveness_score)
            .sum::<f64>()
            / assessments.len() as f64)
    }

    fn identify_controls_gaps(
        &self,
        assessments: &[ControlsAssessmentEntry],
    ) -> Result<Vec<String>, ComplianceError> {
        Ok(assessments
            .iter()
            .filter(|assessment| assessment.effectiveness_score < 70.0)
            .map(|assessment| format!("{}: {}", assessment.assessor_id, assessment.summary))
            .collect())
    }

    fn analyze_compensating_controls(
        &self,
        assessments: &[ControlsAssessmentEntry],
    ) -> Result<Vec<String>, ComplianceError> {
        Ok(assessments
            .iter()
            .filter(|assessment| (50.0..70.0).contains(&assessment.effectiveness_score))
            .map(|assessment| {
                format!(
                    "{}: compensating control recommended",
                    assessment.assessor_id
                )
            })
            .collect())
    }

    fn assess_controls_maturity(
        &self,
        assessments: &[ControlsAssessmentEntry],
    ) -> Result<f64, ComplianceError> {
        self.calculate_overall_controls_effectiveness(assessments)
    }

    fn consolidate_gaps(
        &self,
        analyses: &[GapAnalysisEntry],
    ) -> Result<Vec<ComplianceRiskItem>, ComplianceError> {
        Ok(analyses
            .iter()
            .map(|analysis| ComplianceRiskItem {
                risk_id: format!("GAP-{}", analysis.analyzer_id),
                description: analysis.summary.clone(),
                severity: analysis.severity.clone(),
                related_framework: "General".to_string(),
            })
            .collect())
    }

    fn prioritize_gaps(
        &self,
        gaps: &[ComplianceRiskItem],
    ) -> Result<Vec<ComplianceRiskItem>, ComplianceError> {
        let mut prioritized = gaps.to_vec();
        prioritized.sort_by_key(|item| std::cmp::Reverse(risk_severity_rank(&item.severity)));
        Ok(prioritized)
    }

    fn develop_remediation_roadmap(
        &self,
        prioritized_gaps: &[ComplianceRiskItem],
    ) -> Result<Vec<String>, ComplianceError> {
        Ok(prioritized_gaps
            .iter()
            .map(|gap| format!("Remediate {}: {}", gap.risk_id, gap.description))
            .collect())
    }

    fn analyze_cost_impact(&self, remediation_roadmap: &[String]) -> Result<f64, ComplianceError> {
        Ok(remediation_roadmap.len() as f64 * 5_000.0)
    }

    fn calculate_overall_certification_readiness(
        &self,
        assessments: &[CertificationReadinessAssessment],
    ) -> Result<f64, ComplianceError> {
        if assessments.is_empty() {
            return Ok(0.0);
        }
        Ok(assessments
            .iter()
            .map(|assessment| assessment.readiness_score)
            .sum::<f64>()
            / assessments.len() as f64)
    }

    fn estimate_certification_timeline(
        &self,
        assessments: &[CertificationReadinessAssessment],
    ) -> Result<Duration, ComplianceError> {
        let average_readiness = self.calculate_overall_certification_readiness(assessments)?;
        let months = if average_readiness >= 80.0 {
            2
        } else if average_readiness >= 50.0 {
            6
        } else {
            12
        };
        Ok(Duration::from_secs(86400 * 30 * months))
    }

    fn identify_preparation_requirements(
        &self,
        assessments: &[CertificationReadinessAssessment],
    ) -> Result<Vec<String>, ComplianceError> {
        Ok(assessments
            .iter()
            .filter(|assessment| assessment.readiness_score < 80.0)
            .map(|assessment| {
                format!(
                    "{:?} certification requires additional preparation ({}% ready)",
                    assessment.certification_type, assessment.readiness_score as u32
                )
            })
            .collect())
    }
}

/// Shared `items_met`/`items_total` -> [`ComplianceStatus`] reduction used by both the
/// regulatory-framework and security-standard coverage assessments above.
fn compliance_status_from_coverage(items_met: u32, items_total: u32) -> ComplianceStatus {
    if items_total == 0 {
        ComplianceStatus::NotApplicable
    } else if items_met == items_total {
        ComplianceStatus::Compliant
    } else if items_met == 0 {
        ComplianceStatus::NonCompliant
    } else {
        ComplianceStatus::PartiallyCompliant
    }
}

impl Default for ComplianceFrameworkManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Rank a [`RiskSeverity`] for sorting purposes (higher rank = more severe). `RiskSeverity` does
/// not derive `Ord` (it intentionally has no single global "severity order" semantics elsewhere
/// in the security-analysis framework), so callers that need a total order build one locally.
fn risk_severity_rank(severity: &RiskSeverity) -> u8 {
    match severity {
        RiskSeverity::Low => 1,
        RiskSeverity::Medium => 2,
        RiskSeverity::High => 3,
        RiskSeverity::Critical => 4,
    }
}

impl ComplianceEngine {
    pub fn new_nist() -> Self {
        Self {
            engine_id: "nist_engine".to_string(),
            framework_type: ComplianceFrameworkType::NIST,
            compliance_checkers: Self::initialize_nist_checkers(),
            evidence_collectors: Self::initialize_nist_evidence_collectors(),
            assessment_tools: Self::initialize_nist_assessment_tools(),
            validation_rules: Self::initialize_nist_validation_rules(),
            compliance_metrics: ComplianceMetrics::new_nist(),
            automated_testing: AutomatedComplianceTesting::new_nist(),
            continuous_monitoring: ContinuousComplianceMonitoring::new_nist(),
        }
    }

    pub fn new_gdpr() -> Self {
        Self {
            engine_id: "gdpr_engine".to_string(),
            framework_type: ComplianceFrameworkType::GDPR,
            compliance_checkers: Self::initialize_gdpr_checkers(),
            evidence_collectors: Self::initialize_gdpr_evidence_collectors(),
            assessment_tools: Self::initialize_gdpr_assessment_tools(),
            validation_rules: Self::initialize_gdpr_validation_rules(),
            compliance_metrics: ComplianceMetrics::new_gdpr(),
            automated_testing: AutomatedComplianceTesting::new_gdpr(),
            continuous_monitoring: ContinuousComplianceMonitoring::new_gdpr(),
        }
    }

    pub fn new_hipaa() -> Self {
        Self {
            engine_id: "hipaa_engine".to_string(),
            framework_type: ComplianceFrameworkType::HIPAA,
            compliance_checkers: Self::initialize_hipaa_checkers(),
            evidence_collectors: Self::initialize_hipaa_evidence_collectors(),
            assessment_tools: Self::initialize_hipaa_assessment_tools(),
            validation_rules: Self::initialize_hipaa_validation_rules(),
            compliance_metrics: ComplianceMetrics::new_hipaa(),
            automated_testing: AutomatedComplianceTesting::new_hipaa(),
            continuous_monitoring: ContinuousComplianceMonitoring::new_hipaa(),
        }
    }

    pub fn new_soc2() -> Self {
        Self {
            engine_id: "soc2_engine".to_string(),
            framework_type: ComplianceFrameworkType::SOC2,
            compliance_checkers: Self::initialize_soc2_checkers(),
            evidence_collectors: Self::initialize_soc2_evidence_collectors(),
            assessment_tools: Self::initialize_soc2_assessment_tools(),
            validation_rules: Self::initialize_soc2_validation_rules(),
            compliance_metrics: ComplianceMetrics::new_soc2(),
            automated_testing: AutomatedComplianceTesting::new_soc2(),
            continuous_monitoring: ContinuousComplianceMonitoring::new_soc2(),
        }
    }

    pub fn new_iso27001() -> Self {
        Self {
            engine_id: "iso27001_engine".to_string(),
            framework_type: ComplianceFrameworkType::ISO27001,
            compliance_checkers: Self::initialize_iso27001_checkers(),
            evidence_collectors: Self::initialize_iso27001_evidence_collectors(),
            assessment_tools: Self::initialize_iso27001_assessment_tools(),
            validation_rules: Self::initialize_iso27001_validation_rules(),
            compliance_metrics: ComplianceMetrics::new_iso27001(),
            automated_testing: AutomatedComplianceTesting::new_iso27001(),
            continuous_monitoring: ContinuousComplianceMonitoring::new_iso27001(),
        }
    }

    pub fn new_pci_dss() -> Self {
        Self {
            engine_id: "pci_dss_engine".to_string(),
            framework_type: ComplianceFrameworkType::PciDss,
            compliance_checkers: Self::initialize_pci_dss_checkers(),
            evidence_collectors: Self::initialize_pci_dss_evidence_collectors(),
            assessment_tools: Self::initialize_pci_dss_assessment_tools(),
            validation_rules: Self::initialize_pci_dss_validation_rules(),
            compliance_metrics: ComplianceMetrics::new_pci_dss(),
            automated_testing: AutomatedComplianceTesting::new_pci_dss(),
            continuous_monitoring: ContinuousComplianceMonitoring::new_pci_dss(),
        }
    }

    pub fn assess_framework_compliance(
        &self,
        context: &TraitUsageContext,
    ) -> Result<FrameworkAssessmentResult, ComplianceError> {
        let compliance_status = self.determine_compliance_status(context)?;
        let requirement_assessments = self.assess_requirements(context)?;
        let control_assessments = self.assess_controls(context)?;
        let evidence_summary = self.summarize_evidence(context)?;
        let findings = self.identify_findings(context)?;
        let remediation_items = self.identify_remediation_items(&findings)?;
        let compliance_percentage =
            self.calculate_compliance_percentage(&requirement_assessments)?;
        let maturity_score = self.calculate_maturity_score(&control_assessments)?;

        Ok(FrameworkAssessmentResult {
            framework_type: self.framework_type.clone(),
            compliance_status,
            requirement_assessments,
            control_assessments,
            evidence_summary,
            findings,
            remediation_items,
            compliance_percentage,
            maturity_score,
        })
    }

    fn initialize_nist_checkers() -> Vec<ComplianceChecker> {
        vec![
            ComplianceChecker {
                checker_id: "nist_identify".to_string(),
                function_category: "Identify".to_string(),
                subcategories: vec![
                    "ID.AM".to_string(),
                    "ID.BE".to_string(),
                    "ID.GV".to_string(),
                    "ID.RA".to_string(),
                    "ID.RM".to_string(),
                    "ID.SC".to_string(),
                ],
                assessment_methods: vec![
                    "documentation_review".to_string(),
                    "interview".to_string(),
                ],
            },
            ComplianceChecker {
                checker_id: "nist_protect".to_string(),
                function_category: "Protect".to_string(),
                subcategories: vec![
                    "PR.AC".to_string(),
                    "PR.AT".to_string(),
                    "PR.DS".to_string(),
                    "PR.IP".to_string(),
                    "PR.MA".to_string(),
                    "PR.PT".to_string(),
                ],
                assessment_methods: vec![
                    "technical_testing".to_string(),
                    "observation".to_string(),
                ],
            },
        ]
    }

    fn initialize_gdpr_checkers() -> Vec<ComplianceChecker> {
        vec![
            ComplianceChecker {
                checker_id: "gdpr_data_protection".to_string(),
                function_category: "Data Protection".to_string(),
                subcategories: vec![
                    "Article 5".to_string(),
                    "Article 6".to_string(),
                    "Article 7".to_string(),
                    "Article 25".to_string(),
                    "Article 32".to_string(),
                ],
                assessment_methods: vec![
                    "policy_review".to_string(),
                    "technical_assessment".to_string(),
                ],
            },
            ComplianceChecker {
                checker_id: "gdpr_individual_rights".to_string(),
                function_category: "Individual Rights".to_string(),
                subcategories: vec![
                    "Article 12".to_string(),
                    "Article 15".to_string(),
                    "Article 16".to_string(),
                    "Article 17".to_string(),
                    "Article 20".to_string(),
                ],
                assessment_methods: vec![
                    "process_review".to_string(),
                    "rights_testing".to_string(),
                ],
            },
        ]
    }

    fn initialize_hipaa_checkers() -> Vec<ComplianceChecker> {
        vec![
            ComplianceChecker {
                checker_id: "hipaa_administrative".to_string(),
                function_category: "Administrative Safeguards".to_string(),
                subcategories: vec![
                    "164.308(a)(1)".to_string(),
                    "164.308(a)(2)".to_string(),
                    "164.308(a)(3)".to_string(),
                    "164.308(a)(4)".to_string(),
                ],
                assessment_methods: vec![
                    "policy_review".to_string(),
                    "workforce_training_review".to_string(),
                ],
            },
            ComplianceChecker {
                checker_id: "hipaa_physical".to_string(),
                function_category: "Physical Safeguards".to_string(),
                subcategories: vec![
                    "164.310(a)(1)".to_string(),
                    "164.310(a)(2)".to_string(),
                    "164.310(b)".to_string(),
                    "164.310(c)".to_string(),
                ],
                assessment_methods: vec![
                    "physical_inspection".to_string(),
                    "access_log_review".to_string(),
                ],
            },
        ]
    }

    fn initialize_soc2_checkers() -> Vec<ComplianceChecker> {
        vec![
            ComplianceChecker {
                checker_id: "soc2_security".to_string(),
                function_category: "Security".to_string(),
                subcategories: vec![
                    "CC6.1".to_string(),
                    "CC6.2".to_string(),
                    "CC6.3".to_string(),
                    "CC6.6".to_string(),
                    "CC6.7".to_string(),
                    "CC6.8".to_string(),
                ],
                assessment_methods: vec!["control_testing".to_string(), "inquiry".to_string()],
            },
            ComplianceChecker {
                checker_id: "soc2_availability".to_string(),
                function_category: "Availability".to_string(),
                subcategories: vec!["A1.1".to_string(), "A1.2".to_string(), "A1.3".to_string()],
                assessment_methods: vec![
                    "performance_monitoring".to_string(),
                    "incident_review".to_string(),
                ],
            },
        ]
    }

    fn initialize_iso27001_checkers() -> Vec<ComplianceChecker> {
        vec![
            ComplianceChecker {
                checker_id: "iso27001_isms".to_string(),
                function_category: "ISMS".to_string(),
                subcategories: vec![
                    "A.5".to_string(),
                    "A.6".to_string(),
                    "A.7".to_string(),
                    "A.8".to_string(),
                    "A.9".to_string(),
                ],
                assessment_methods: vec![
                    "documentation_review".to_string(),
                    "management_interview".to_string(),
                ],
            },
            ComplianceChecker {
                checker_id: "iso27001_technical".to_string(),
                function_category: "Technical Controls".to_string(),
                subcategories: vec![
                    "A.12".to_string(),
                    "A.13".to_string(),
                    "A.14".to_string(),
                    "A.15".to_string(),
                    "A.16".to_string(),
                    "A.17".to_string(),
                ],
                assessment_methods: vec![
                    "technical_testing".to_string(),
                    "configuration_review".to_string(),
                ],
            },
        ]
    }

    fn initialize_pci_dss_checkers() -> Vec<ComplianceChecker> {
        vec![
            ComplianceChecker {
                checker_id: "pci_dss_network".to_string(),
                function_category: "Network Security".to_string(),
                subcategories: vec!["Requirement 1".to_string(), "Requirement 2".to_string()],
                assessment_methods: vec![
                    "network_scan".to_string(),
                    "configuration_review".to_string(),
                ],
            },
            ComplianceChecker {
                checker_id: "pci_dss_data_protection".to_string(),
                function_category: "Data Protection".to_string(),
                subcategories: vec!["Requirement 3".to_string(), "Requirement 4".to_string()],
                assessment_methods: vec![
                    "data_flow_analysis".to_string(),
                    "encryption_testing".to_string(),
                ],
            },
        ]
    }
}

impl ComplianceEngine {
    fn initialize_nist_evidence_collectors() -> Vec<EvidenceCollector> {
        vec![EvidenceCollector {
            collector_id: "nist_ec".to_string(),
            evidence_types: vec!["documentation".to_string(), "logs".to_string()],
            collection_methods: vec!["automated_scan".to_string()],
        }]
    }
    fn initialize_nist_assessment_tools() -> Vec<AssessmentTool> {
        vec![AssessmentTool {
            tool_id: "nist_at".to_string(),
            tool_type: "framework_mapper".to_string(),
            capabilities: vec!["gap_analysis".to_string()],
        }]
    }
    fn initialize_nist_validation_rules() -> Vec<ValidationRule> {
        vec![ValidationRule {
            rule_id: "nist_vr".to_string(),
            rule_description: "Identify function must be documented".to_string(),
            validation_criteria: vec!["asset_inventory".to_string()],
        }]
    }

    fn initialize_gdpr_evidence_collectors() -> Vec<EvidenceCollector> {
        vec![EvidenceCollector {
            collector_id: "gdpr_ec".to_string(),
            evidence_types: vec!["consent_records".to_string(), "processing_logs".to_string()],
            collection_methods: vec!["policy_review".to_string()],
        }]
    }
    fn initialize_gdpr_assessment_tools() -> Vec<AssessmentTool> {
        vec![AssessmentTool {
            tool_id: "gdpr_at".to_string(),
            tool_type: "privacy_scanner".to_string(),
            capabilities: vec!["data_flow_mapping".to_string()],
        }]
    }
    fn initialize_gdpr_validation_rules() -> Vec<ValidationRule> {
        vec![ValidationRule {
            rule_id: "gdpr_vr".to_string(),
            rule_description: "Personal data must have a lawful basis".to_string(),
            validation_criteria: vec!["consent_or_lawful_basis".to_string()],
        }]
    }

    fn initialize_hipaa_evidence_collectors() -> Vec<EvidenceCollector> {
        vec![EvidenceCollector {
            collector_id: "hipaa_ec".to_string(),
            evidence_types: vec!["access_logs".to_string(), "training_records".to_string()],
            collection_methods: vec!["log_review".to_string()],
        }]
    }
    fn initialize_hipaa_assessment_tools() -> Vec<AssessmentTool> {
        vec![AssessmentTool {
            tool_id: "hipaa_at".to_string(),
            tool_type: "safeguard_checker".to_string(),
            capabilities: vec!["access_control_review".to_string()],
        }]
    }
    fn initialize_hipaa_validation_rules() -> Vec<ValidationRule> {
        vec![ValidationRule {
            rule_id: "hipaa_vr".to_string(),
            rule_description: "PHI must be protected by administrative safeguards".to_string(),
            validation_criteria: vec!["workforce_training".to_string()],
        }]
    }

    fn initialize_soc2_evidence_collectors() -> Vec<EvidenceCollector> {
        vec![EvidenceCollector {
            collector_id: "soc2_ec".to_string(),
            evidence_types: vec!["control_test_results".to_string()],
            collection_methods: vec!["control_testing".to_string()],
        }]
    }
    fn initialize_soc2_assessment_tools() -> Vec<AssessmentTool> {
        vec![AssessmentTool {
            tool_id: "soc2_at".to_string(),
            tool_type: "trust_services_mapper".to_string(),
            capabilities: vec!["criteria_mapping".to_string()],
        }]
    }
    fn initialize_soc2_validation_rules() -> Vec<ValidationRule> {
        vec![ValidationRule {
            rule_id: "soc2_vr".to_string(),
            rule_description: "Security criteria must have documented controls".to_string(),
            validation_criteria: vec!["control_documentation".to_string()],
        }]
    }

    fn initialize_iso27001_evidence_collectors() -> Vec<EvidenceCollector> {
        vec![EvidenceCollector {
            collector_id: "iso27001_ec".to_string(),
            evidence_types: vec!["isms_documentation".to_string()],
            collection_methods: vec!["documentation_review".to_string()],
        }]
    }
    fn initialize_iso27001_assessment_tools() -> Vec<AssessmentTool> {
        vec![AssessmentTool {
            tool_id: "iso27001_at".to_string(),
            tool_type: "control_mapper".to_string(),
            capabilities: vec!["annex_a_mapping".to_string()],
        }]
    }
    fn initialize_iso27001_validation_rules() -> Vec<ValidationRule> {
        vec![ValidationRule {
            rule_id: "iso27001_vr".to_string(),
            rule_description: "ISMS scope must be documented".to_string(),
            validation_criteria: vec!["scope_statement".to_string()],
        }]
    }

    fn initialize_pci_dss_evidence_collectors() -> Vec<EvidenceCollector> {
        vec![EvidenceCollector {
            collector_id: "pci_dss_ec".to_string(),
            evidence_types: vec!["network_diagrams".to_string(), "scan_reports".to_string()],
            collection_methods: vec!["network_scan".to_string()],
        }]
    }
    fn initialize_pci_dss_assessment_tools() -> Vec<AssessmentTool> {
        vec![AssessmentTool {
            tool_id: "pci_dss_at".to_string(),
            tool_type: "vulnerability_scanner".to_string(),
            capabilities: vec!["cardholder_data_discovery".to_string()],
        }]
    }
    fn initialize_pci_dss_validation_rules() -> Vec<ValidationRule> {
        vec![ValidationRule {
            rule_id: "pci_dss_vr".to_string(),
            rule_description: "Cardholder data must be encrypted in transit".to_string(),
            validation_criteria: vec!["encryption_in_transit".to_string()],
        }]
    }

    fn determine_compliance_status(
        &self,
        context: &TraitUsageContext,
    ) -> Result<ComplianceStatus, ComplianceError> {
        let rule_coverage =
            self.validation_rules.len() + self.continuous_monitoring.monitoring_rules.len();
        let status = if context.handles_personal_data && !context.has_data_anonymization {
            ComplianceStatus::NonCompliant
        } else if !context.has_audit_logging || !context.has_access_controls {
            ComplianceStatus::PartiallyCompliant
        } else if rule_coverage == 0 {
            ComplianceStatus::NotAssessed
        } else {
            ComplianceStatus::Compliant
        };
        Ok(status)
    }

    fn assess_requirements(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<RequirementAssessment>, ComplianceError> {
        Ok(self
            .compliance_checkers
            .iter()
            .flat_map(|checker| {
                checker
                    .subcategories
                    .iter()
                    .map(move |subcategory| RequirementAssessment {
                        requirement_id: subcategory.clone(),
                        category: checker.function_category.clone(),
                        status: if context.has_audit_logging {
                            ComplianceStatus::Compliant
                        } else {
                            ComplianceStatus::PartiallyCompliant
                        },
                    })
            })
            .collect())
    }

    fn assess_controls(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ControlAssessment>, ComplianceError> {
        Ok(self
            .assessment_tools
            .iter()
            .map(|tool| ControlAssessment {
                control_id: tool.tool_id.clone(),
                control_type: tool.tool_type.clone(),
                effective: context.has_access_controls || context.has_encryption,
            })
            .collect())
    }

    fn summarize_evidence(
        &self,
        _context: &TraitUsageContext,
    ) -> Result<EvidenceSummary, ComplianceError> {
        Ok(EvidenceSummary {
            total_evidence_items: self.evidence_collectors.len() as u32,
            evidence_types: self
                .evidence_collectors
                .iter()
                .flat_map(|collector| collector.evidence_types.clone())
                .collect(),
            collection_methods: self
                .evidence_collectors
                .iter()
                .flat_map(|collector| collector.collection_methods.clone())
                .collect(),
        })
    }

    fn identify_findings(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ComplianceFinding>, ComplianceError> {
        let mut findings = Vec::new();
        if context.handles_personal_data && !context.has_data_anonymization {
            findings.push(ComplianceFinding {
                finding_id: format!("{}-FIND-001", self.engine_id),
                description: "Personal data processed without anonymization".to_string(),
                severity: RiskSeverity::High,
            });
        }
        if !context.has_audit_logging {
            findings.push(ComplianceFinding {
                finding_id: format!("{}-FIND-002", self.engine_id),
                description: "Audit logging is not enabled".to_string(),
                severity: RiskSeverity::Medium,
            });
        }
        if self.compliance_metrics.metric_definitions.is_empty() {
            findings.push(ComplianceFinding {
                finding_id: format!("{}-FIND-003", self.engine_id),
                description: "No compliance metrics are defined for this framework".to_string(),
                severity: RiskSeverity::Low,
            });
        }
        Ok(findings)
    }

    fn identify_remediation_items(
        &self,
        findings: &[ComplianceFinding],
    ) -> Result<Vec<RemediationItem>, ComplianceError> {
        Ok(findings
            .iter()
            .map(|finding| RemediationItem {
                item_id: format!("REM-{}", finding.finding_id),
                description: finding.description.clone(),
                priority: match finding.severity {
                    RiskSeverity::Critical => MitigationPriority::Critical,
                    RiskSeverity::High => MitigationPriority::High,
                    RiskSeverity::Medium => MitigationPriority::Medium,
                    RiskSeverity::Low => MitigationPriority::Low,
                },
                estimated_effort: ImplementationEffort::Medium,
            })
            .collect())
    }

    fn calculate_compliance_percentage(
        &self,
        requirement_assessments: &[RequirementAssessment],
    ) -> Result<f64, ComplianceError> {
        if requirement_assessments.is_empty() {
            return Ok(100.0);
        }
        let compliant = requirement_assessments
            .iter()
            .filter(|assessment| assessment.status == ComplianceStatus::Compliant)
            .count();
        Ok(100.0 * compliant as f64 / requirement_assessments.len() as f64)
    }

    fn calculate_maturity_score(
        &self,
        control_assessments: &[ControlAssessment],
    ) -> Result<f64, ComplianceError> {
        if control_assessments.is_empty() {
            return Ok(0.0);
        }
        let effective = control_assessments
            .iter()
            .filter(|assessment| assessment.effective)
            .count();
        let base_score = 10.0 * effective as f64 / control_assessments.len() as f64;
        let automation_bonus = if self.automated_testing.test_suites.is_empty() {
            0.0
        } else {
            0.5
        };
        Ok((base_score + automation_bonus).min(10.0))
    }
}

impl ComplianceMetrics {
    pub fn new_nist() -> Self {
        Self {
            metric_definitions: vec![MetricDefinition {
                metric_name: "nist_csf_coverage".to_string(),
                description: "NIST CSF subcategory coverage".to_string(),
                calculation_method: "addressed / total".to_string(),
            }],
            measurement_methods: vec!["automated_scan".to_string()],
        }
    }
    pub fn new_gdpr() -> Self {
        Self {
            metric_definitions: vec![MetricDefinition {
                metric_name: "gdpr_article_coverage".to_string(),
                description: "GDPR article coverage".to_string(),
                calculation_method: "addressed / total".to_string(),
            }],
            measurement_methods: vec!["policy_review".to_string()],
        }
    }
    pub fn new_hipaa() -> Self {
        Self {
            metric_definitions: vec![MetricDefinition {
                metric_name: "hipaa_safeguard_coverage".to_string(),
                description: "HIPAA safeguard coverage".to_string(),
                calculation_method: "addressed / total".to_string(),
            }],
            measurement_methods: vec!["log_review".to_string()],
        }
    }
    pub fn new_soc2() -> Self {
        Self {
            metric_definitions: vec![MetricDefinition {
                metric_name: "soc2_criteria_coverage".to_string(),
                description: "SOC 2 trust services criteria coverage".to_string(),
                calculation_method: "addressed / total".to_string(),
            }],
            measurement_methods: vec!["control_testing".to_string()],
        }
    }
    pub fn new_iso27001() -> Self {
        Self {
            metric_definitions: vec![MetricDefinition {
                metric_name: "iso27001_annex_a_coverage".to_string(),
                description: "ISO 27001 Annex A control coverage".to_string(),
                calculation_method: "addressed / total".to_string(),
            }],
            measurement_methods: vec!["documentation_review".to_string()],
        }
    }
    pub fn new_pci_dss() -> Self {
        Self {
            metric_definitions: vec![MetricDefinition {
                metric_name: "pci_dss_requirement_coverage".to_string(),
                description: "PCI DSS requirement coverage".to_string(),
                calculation_method: "addressed / total".to_string(),
            }],
            measurement_methods: vec!["network_scan".to_string()],
        }
    }
}

impl AutomatedComplianceTesting {
    pub fn new_nist() -> Self {
        Self {
            test_suites: vec![TestSuite {
                suite_name: "nist_csf_tests".to_string(),
                test_cases: vec![TestCase {
                    test_name: "asset_inventory_present".to_string(),
                    test_description: "Verify asset inventory exists".to_string(),
                    expected_result: "pass".to_string(),
                }],
            }],
            test_schedule: TestSchedule {
                frequency: Duration::from_secs(86400 * 30),
                next_execution: SystemTime::now(),
            },
        }
    }
    pub fn new_gdpr() -> Self {
        Self {
            test_suites: vec![TestSuite {
                suite_name: "gdpr_tests".to_string(),
                test_cases: vec![TestCase {
                    test_name: "lawful_basis_documented".to_string(),
                    test_description: "Verify lawful basis is documented".to_string(),
                    expected_result: "pass".to_string(),
                }],
            }],
            test_schedule: TestSchedule {
                frequency: Duration::from_secs(86400 * 30),
                next_execution: SystemTime::now(),
            },
        }
    }
    pub fn new_hipaa() -> Self {
        Self {
            test_suites: vec![TestSuite {
                suite_name: "hipaa_tests".to_string(),
                test_cases: vec![TestCase {
                    test_name: "access_controls_enforced".to_string(),
                    test_description: "Verify access controls are enforced".to_string(),
                    expected_result: "pass".to_string(),
                }],
            }],
            test_schedule: TestSchedule {
                frequency: Duration::from_secs(86400 * 90),
                next_execution: SystemTime::now(),
            },
        }
    }
    pub fn new_soc2() -> Self {
        Self {
            test_suites: vec![TestSuite {
                suite_name: "soc2_tests".to_string(),
                test_cases: vec![TestCase {
                    test_name: "security_controls_tested".to_string(),
                    test_description: "Verify security controls are tested".to_string(),
                    expected_result: "pass".to_string(),
                }],
            }],
            test_schedule: TestSchedule {
                frequency: Duration::from_secs(86400 * 365),
                next_execution: SystemTime::now(),
            },
        }
    }
    pub fn new_iso27001() -> Self {
        Self {
            test_suites: vec![TestSuite {
                suite_name: "iso27001_tests".to_string(),
                test_cases: vec![TestCase {
                    test_name: "isms_scope_documented".to_string(),
                    test_description: "Verify ISMS scope is documented".to_string(),
                    expected_result: "pass".to_string(),
                }],
            }],
            test_schedule: TestSchedule {
                frequency: Duration::from_secs(86400 * 90),
                next_execution: SystemTime::now(),
            },
        }
    }
    pub fn new_pci_dss() -> Self {
        Self {
            test_suites: vec![TestSuite {
                suite_name: "pci_dss_tests".to_string(),
                test_cases: vec![TestCase {
                    test_name: "cardholder_data_encrypted".to_string(),
                    test_description: "Verify cardholder data is encrypted in transit".to_string(),
                    expected_result: "pass".to_string(),
                }],
            }],
            test_schedule: TestSchedule {
                frequency: Duration::from_secs(86400 * 30),
                next_execution: SystemTime::now(),
            },
        }
    }
}

impl ContinuousComplianceMonitoring {
    pub fn new_nist() -> Self {
        Self {
            monitoring_rules: vec![MonitoringRule {
                rule_name: "nist_control_drift".to_string(),
                condition: "control_effectiveness < 0.7".to_string(),
                action: "alert".to_string(),
            }],
            alert_thresholds: vec![AlertThreshold {
                threshold_name: "nist_compliance_floor".to_string(),
                threshold_value: 0.7,
                alert_level: AlertLevel::High,
            }],
        }
    }
    pub fn new_gdpr() -> Self {
        Self {
            monitoring_rules: vec![MonitoringRule {
                rule_name: "gdpr_consent_expiry".to_string(),
                condition: "consent_age > retention_period".to_string(),
                action: "alert".to_string(),
            }],
            alert_thresholds: vec![AlertThreshold {
                threshold_name: "gdpr_compliance_floor".to_string(),
                threshold_value: 0.9,
                alert_level: AlertLevel::Critical,
            }],
        }
    }
    pub fn new_hipaa() -> Self {
        Self {
            monitoring_rules: vec![MonitoringRule {
                rule_name: "hipaa_access_anomaly".to_string(),
                condition: "unauthorized_access_attempt".to_string(),
                action: "alert".to_string(),
            }],
            alert_thresholds: vec![AlertThreshold {
                threshold_name: "hipaa_compliance_floor".to_string(),
                threshold_value: 0.9,
                alert_level: AlertLevel::Critical,
            }],
        }
    }
    pub fn new_soc2() -> Self {
        Self {
            monitoring_rules: vec![MonitoringRule {
                rule_name: "soc2_control_failure".to_string(),
                condition: "control_test_failed".to_string(),
                action: "alert".to_string(),
            }],
            alert_thresholds: vec![AlertThreshold {
                threshold_name: "soc2_compliance_floor".to_string(),
                threshold_value: 0.8,
                alert_level: AlertLevel::High,
            }],
        }
    }
    pub fn new_iso27001() -> Self {
        Self {
            monitoring_rules: vec![MonitoringRule {
                rule_name: "iso27001_nonconformity".to_string(),
                condition: "audit_finding_open".to_string(),
                action: "alert".to_string(),
            }],
            alert_thresholds: vec![AlertThreshold {
                threshold_name: "iso27001_compliance_floor".to_string(),
                threshold_value: 0.8,
                alert_level: AlertLevel::Medium,
            }],
        }
    }
    pub fn new_pci_dss() -> Self {
        Self {
            monitoring_rules: vec![MonitoringRule {
                rule_name: "pci_dss_scan_failure".to_string(),
                condition: "vulnerability_scan_failed".to_string(),
                action: "alert".to_string(),
            }],
            alert_thresholds: vec![AlertThreshold {
                threshold_name: "pci_dss_compliance_floor".to_string(),
                threshold_value: 0.95,
                alert_level: AlertLevel::Critical,
            }],
        }
    }
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

impl SecurityStandard {
    pub fn new_iso27001() -> Self {
        Self {
            standard_id: "ISO27001".to_string(),
            name: "ISO/IEC 27001".to_string(),
            organization: "ISO/IEC".to_string(),
            version: "2022".to_string(),
            publication_date: SystemTime::now(),
            standard_type: SecurityStandardType::Management,
            security_objectives: vec![SecurityObjective {
                objective_id: "A.5".to_string(),
                description: "Information security policies".to_string(),
            }],
            security_controls: vec![SecurityControl {
                control_id: "A.8".to_string(),
                name: "Asset management".to_string(),
                category: "Operational".to_string(),
            }],
            implementation_guidance: vec![ImplementationGuide {
                guide_id: "ISO-G1".to_string(),
                title: "Establish an information security management system".to_string(),
            }],
            assessment_criteria: vec![AssessmentCriterion {
                criterion_id: "ISO-AC1".to_string(),
                description: "ISMS scope is documented".to_string(),
                weight: 1.0,
            }],
            maturity_levels: vec![MaturityLevel {
                level: 3,
                name: "Defined".to_string(),
            }],
            cross_references: HashMap::new(),
        }
    }

    pub fn new_nist_csf() -> Self {
        Self {
            standard_id: "NIST_CSF".to_string(),
            name: "NIST Cybersecurity Framework".to_string(),
            organization: "NIST".to_string(),
            version: "2.0".to_string(),
            publication_date: SystemTime::now(),
            standard_type: SecurityStandardType::Technical,
            security_objectives: vec![SecurityObjective {
                objective_id: "GV".to_string(),
                description: "Govern organizational cybersecurity risk".to_string(),
            }],
            security_controls: vec![SecurityControl {
                control_id: "PR.AC".to_string(),
                name: "Identity management and access control".to_string(),
                category: "Protect".to_string(),
            }],
            implementation_guidance: vec![ImplementationGuide {
                guide_id: "CSF-G1".to_string(),
                title: "Adopt a risk-based cybersecurity program".to_string(),
            }],
            assessment_criteria: vec![AssessmentCriterion {
                criterion_id: "CSF-AC1".to_string(),
                description: "Cybersecurity risk is governed at the organizational level"
                    .to_string(),
                weight: 1.0,
            }],
            maturity_levels: vec![MaturityLevel {
                level: 2,
                name: "Risk Informed".to_string(),
            }],
            cross_references: HashMap::new(),
        }
    }

    pub fn new_cobit() -> Self {
        Self {
            standard_id: "COBIT".to_string(),
            name: "Control Objectives for Information and Related Technologies".to_string(),
            organization: "ISACA".to_string(),
            version: "2019".to_string(),
            publication_date: SystemTime::now(),
            standard_type: SecurityStandardType::Management,
            security_objectives: vec![SecurityObjective {
                objective_id: "APO13".to_string(),
                description: "Manage security".to_string(),
            }],
            security_controls: vec![SecurityControl {
                control_id: "DSS05".to_string(),
                name: "Managed security services".to_string(),
                category: "Operational".to_string(),
            }],
            implementation_guidance: vec![ImplementationGuide {
                guide_id: "COBIT-G1".to_string(),
                title: "Align IT governance with enterprise goals".to_string(),
            }],
            assessment_criteria: vec![AssessmentCriterion {
                criterion_id: "COBIT-AC1".to_string(),
                description: "IT governance objectives are documented".to_string(),
                weight: 1.0,
            }],
            maturity_levels: vec![MaturityLevel {
                level: 3,
                name: "Established".to_string(),
            }],
            cross_references: HashMap::new(),
        }
    }

    pub fn new_itil() -> Self {
        Self {
            standard_id: "ITIL".to_string(),
            name: "Information Technology Infrastructure Library".to_string(),
            organization: "AXELOS".to_string(),
            version: "4".to_string(),
            publication_date: SystemTime::now(),
            standard_type: SecurityStandardType::Operational,
            security_objectives: vec![SecurityObjective {
                objective_id: "ITIL-ISM".to_string(),
                description: "Information security management".to_string(),
            }],
            security_controls: vec![SecurityControl {
                control_id: "ITIL-C1".to_string(),
                name: "Incident management".to_string(),
                category: "Operational".to_string(),
            }],
            implementation_guidance: vec![ImplementationGuide {
                guide_id: "ITIL-G1".to_string(),
                title: "Establish a service value system".to_string(),
            }],
            assessment_criteria: vec![AssessmentCriterion {
                criterion_id: "ITIL-AC1".to_string(),
                description: "Incident management process is documented".to_string(),
                weight: 1.0,
            }],
            maturity_levels: vec![MaturityLevel {
                level: 2,
                name: "Repeatable".to_string(),
            }],
            cross_references: HashMap::new(),
        }
    }

    pub fn new_cis_controls() -> Self {
        Self {
            standard_id: "CIS_Controls".to_string(),
            name: "CIS Critical Security Controls".to_string(),
            organization: "Center for Internet Security".to_string(),
            version: "8.1".to_string(),
            publication_date: SystemTime::now(),
            standard_type: SecurityStandardType::Technical,
            security_objectives: vec![SecurityObjective {
                objective_id: "CIS-1".to_string(),
                description: "Inventory and control of enterprise assets".to_string(),
            }],
            security_controls: vec![SecurityControl {
                control_id: "CIS-4".to_string(),
                name: "Secure configuration of enterprise assets".to_string(),
                category: "Technical".to_string(),
            }],
            implementation_guidance: vec![ImplementationGuide {
                guide_id: "CIS-G1".to_string(),
                title: "Implement Implementation Group 1 safeguards first".to_string(),
            }],
            assessment_criteria: vec![AssessmentCriterion {
                criterion_id: "CIS-AC1".to_string(),
                description: "Asset inventory is maintained".to_string(),
                weight: 1.0,
            }],
            maturity_levels: vec![MaturityLevel {
                level: 1,
                name: "IG1".to_string(),
            }],
            cross_references: HashMap::new(),
        }
    }
}

impl AuditManager {
    /// Conduct an audit using this manager's configuration, producing a summary result whose
    /// finding count reflects how many risk-relevant signals are present in `context`.
    pub fn conduct_audit(
        &self,
        context: &TraitUsageContext,
    ) -> Result<AuditResult, ComplianceError> {
        let mut findings_count = 0;
        if context.has_unsafe_operations && !context.has_bounds_checking {
            findings_count += 1;
        }
        if context.handles_sensitive_data && !context.has_encryption {
            findings_count += 1;
        }
        if !context.has_audit_logging {
            findings_count += 1;
        }
        let summary = format!(
            "{:?} audit of '{}' completed with {findings_count} finding(s)",
            self.audit_type, context.trait_name
        );
        Ok(AuditResult {
            audit_id: self.audit_id.clone(),
            audit_type: self.audit_type.clone(),
            findings_count,
            summary,
            conducted_at: SystemTime::now(),
        })
    }
}

impl PolicyEngine {
    pub fn assess_policy_compliance(
        &self,
        context: &TraitUsageContext,
    ) -> Result<PolicyComplianceAssessment, ComplianceError> {
        let status = if !context.has_access_controls {
            ComplianceStatus::NonCompliant
        } else if !context.has_audit_logging {
            ComplianceStatus::PartiallyCompliant
        } else {
            ComplianceStatus::Compliant
        };
        let summary = format!(
            "Policy '{}' evaluated against '{}'",
            self.policy_id, context.trait_name
        );
        Ok(PolicyComplianceAssessment {
            policy_id: self.policy_id.clone(),
            status,
            summary,
        })
    }
}

impl ControlsAssessor {
    pub fn assess_controls(
        &self,
        context: &TraitUsageContext,
    ) -> Result<ControlsAssessmentEntry, ComplianceError> {
        let mut score: f64 = 100.0;
        if !context.has_access_controls {
            score -= 30.0;
        }
        if !context.has_encryption && context.handles_sensitive_data {
            score -= 30.0;
        }
        if !context.has_input_validation {
            score -= 20.0;
        }
        let score = score.max(0.0);
        let summary = format!(
            "Controls assessment for '{}' scored {score:.1}",
            context.trait_name
        );
        Ok(ControlsAssessmentEntry {
            assessor_id: self.assessor_id.clone(),
            effectiveness_score: score,
            summary,
        })
    }
}

impl GapAnalyzer {
    pub fn perform_gap_analysis(
        &self,
        context: &TraitUsageContext,
    ) -> Result<GapAnalysisEntry, ComplianceError> {
        let (summary, severity) =
            if context.handles_personal_data && !context.has_data_anonymization {
                (
                    format!(
                        "'{}' processes personal data without anonymization",
                        context.trait_name
                    ),
                    RiskSeverity::High,
                )
            } else if !context.has_encryption && context.handles_sensitive_data {
                (
                    format!(
                        "'{}' handles sensitive data without encryption",
                        context.trait_name
                    ),
                    RiskSeverity::Critical,
                )
            } else {
                (
                    format!(
                        "No significant gaps identified for '{}'",
                        context.trait_name
                    ),
                    RiskSeverity::Low,
                )
            };
        Ok(GapAnalysisEntry {
            analyzer_id: self.analyzer_id.clone(),
            summary,
            severity,
        })
    }
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
pub enum ComplianceError {
    AssessmentError(String),
    FrameworkError(String),
    RegulatoryError(String),
    AuditError(String),
    PolicyError(String),
    ControlsError(String),
    CertificationError(String),
    DocumentationError(String),
    ConfigurationError(String),
    DataError(String),
}

impl std::fmt::Display for ComplianceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplianceError::AssessmentError(msg) => write!(f, "Assessment error: {}", msg),
            ComplianceError::FrameworkError(msg) => write!(f, "Framework error: {}", msg),
            ComplianceError::RegulatoryError(msg) => write!(f, "Regulatory error: {}", msg),
            ComplianceError::AuditError(msg) => write!(f, "Audit error: {}", msg),
            ComplianceError::PolicyError(msg) => write!(f, "Policy error: {}", msg),
            ComplianceError::ControlsError(msg) => write!(f, "Controls error: {}", msg),
            ComplianceError::CertificationError(msg) => write!(f, "Certification error: {}", msg),
            ComplianceError::DocumentationError(msg) => write!(f, "Documentation error: {}", msg),
            ComplianceError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            ComplianceError::DataError(msg) => write!(f, "Data error: {}", msg),
        }
    }
}

impl std::error::Error for ComplianceError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfiguration {
    pub enabled_frameworks: Vec<ComplianceFrameworkType>,
    pub assessment_frequency: HashMap<String, Duration>,
    pub audit_retention_period: Duration,
    pub automated_monitoring: bool,
    pub real_time_alerting: bool,
    pub compliance_threshold: f64,
    pub evidence_collection_enabled: bool,
    pub continuous_assessment: bool,
}

impl Default for ComplianceConfiguration {
    fn default() -> Self {
        let mut assessment_frequency = HashMap::new();
        assessment_frequency.insert("GDPR".to_string(), Duration::from_secs(86400 * 30)); // Monthly
        assessment_frequency.insert("HIPAA".to_string(), Duration::from_secs(86400 * 90)); // Quarterly
        assessment_frequency.insert("SOC2".to_string(), Duration::from_secs(86400 * 365)); // Annually

        Self {
            enabled_frameworks: vec![
                ComplianceFrameworkType::NIST,
                ComplianceFrameworkType::GDPR,
                ComplianceFrameworkType::HIPAA,
                ComplianceFrameworkType::SOC2,
            ],
            assessment_frequency,
            audit_retention_period: Duration::from_secs(86400 * 365 * 7), // 7 years
            automated_monitoring: true,
            real_time_alerting: true,
            compliance_threshold: 0.85,
            evidence_collection_enabled: true,
            continuous_assessment: true,
        }
    }
}

macro_rules! define_compliance_supporting_types {
    () => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ComplianceChecker {
            pub checker_id: String,
            pub function_category: String,
            pub subcategories: Vec<String>,
            pub assessment_methods: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct EvidenceCollector {
            pub collector_id: String,
            pub evidence_types: Vec<String>,
            pub collection_methods: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct AssessmentTool {
            pub tool_id: String,
            pub tool_type: String,
            pub capabilities: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ValidationRule {
            pub rule_id: String,
            pub rule_description: String,
            pub validation_criteria: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ComplianceMetrics {
            pub metric_definitions: Vec<MetricDefinition>,
            pub measurement_methods: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct MetricDefinition {
            pub metric_name: String,
            pub description: String,
            pub calculation_method: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct AutomatedComplianceTesting {
            pub test_suites: Vec<TestSuite>,
            pub test_schedule: TestSchedule,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct TestSuite {
            pub suite_name: String,
            pub test_cases: Vec<TestCase>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct TestCase {
            pub test_name: String,
            pub test_description: String,
            pub expected_result: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct TestSchedule {
            pub frequency: Duration,
            pub next_execution: SystemTime,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ContinuousComplianceMonitoring {
            pub monitoring_rules: Vec<MonitoringRule>,
            pub alert_thresholds: Vec<AlertThreshold>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct MonitoringRule {
            pub rule_name: String,
            pub condition: String,
            pub action: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct AlertThreshold {
            pub threshold_name: String,
            pub threshold_value: f64,
            pub alert_level: AlertLevel,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum AlertLevel {
            Low,
            Medium,
            High,
            Critical,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct CachedComplianceResult {
            pub result: ComplianceAssessmentResult,
            pub cache_timestamp: SystemTime,
            pub cache_ttl: Duration,
        }
    };
}

define_compliance_supporting_types!();

pub fn create_compliance_framework_manager() -> ComplianceFrameworkManager {
    ComplianceFrameworkManager::new()
}

pub fn assess_comprehensive_compliance(
    context: &TraitUsageContext,
) -> Result<ComplianceAssessmentResult, ComplianceError> {
    let mut manager = ComplianceFrameworkManager::new();
    manager.assess_compliance(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn high_risk_context() -> TraitUsageContext {
        TraitUsageContext {
            trait_name: "Serialize".to_string(),
            traits: vec!["Serialize".to_string()],
            handles_sensitive_data: true,
            handles_personal_data: true,
            has_audit_logging: false,
            has_access_controls: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_assess_compliance_and_check_status_smoke() {
        let mut manager = create_compliance_framework_manager();
        let context = high_risk_context();

        let assessment = manager
            .assess_compliance(&context)
            .expect("assessment should succeed");
        assert!(!assessment.framework_assessments.is_empty());
        assert!((0.0..=10.0).contains(&assessment.compliance_score));
        assert!(!assessment.risk_assessment.identified_risks.is_empty());

        assert!(manager.check_compliance_status(&context).is_ok());

        let default_manager = ComplianceFrameworkManager::default();
        assert!(default_manager.compliance_monitors().is_empty());
    }
}
