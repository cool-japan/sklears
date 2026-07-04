//! Pure-data support types for [`super::ThreatModelingEngine`] and friends.
//!
//! Everything in this file is either a plain data container (struct/enum with no
//! significant logic) or a small piece of self-contained, per-type logic (the six STRIDE
//! `*Detector::detect_*` methods, `ThreatIntelligenceSource::gather_insights`, and the
//! trivial `new()`/`Default` pairs for the handful of inert configuration containers used
//! by [`super::AttackTreeGenerator`] / [`super::ThreatIntelligenceManager`] /
//! [`super::ThreatLandscapeAssessment`]). The actual context-driven heuristics live on
//! [`super::ThreatModelingEngine`] in the parent module, which needs private-field access
//! to `ThreatModelingEngine`/`StrideAnalyzer`/`AttackTreeGenerator`/
//! `ThreatIntelligenceManager`/`ThreatLandscapeAssessment` and therefore cannot live here.
//!
//! Split out of `threat_modeling.rs` (which was approaching the workspace's 2000-line
//! refactor threshold) via the `splitrs`-style types/logic separation used elsewhere in
//! this crate.

use super::super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
// `ThreatLandscapeAssessment` keeps private fields and therefore stays defined in the
// parent `threat_modeling` module (its fields are populated via struct-literal
// construction in `ThreatModelingEngine::assess_threat_landscape`); it is only *named*
// here as a field type on `ThreatModelingResult`, which does not require field access.
use super::ThreatLandscapeAssessment;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StrideCategory {
    Spoofing,
    Tampering,
    Repudiation,
    InformationDisclosure,
    DenialOfService,
    ElevationOfPrivilege,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatScenario {
    pub scenario_id: String,
    pub name: String,
    pub description: String,
    pub attack_vectors: Vec<String>,
    pub threat_actors: Vec<ThreatActor>,
    pub assets_at_risk: Vec<String>,
    pub impact_assessment: ImpactAssessment,
    pub likelihood: f64,
    pub detection_methods: Vec<DetectionMethod>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
    pub timeline: ThreatTimeline,
    pub scenario_variants: Vec<ScenarioVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackVector {
    pub vector_id: String,
    pub name: String,
    pub description: String,
    pub attack_surface: AttackSurface,
    pub entry_points: Vec<EntryPoint>,
    pub prerequisites: Vec<String>,
    pub attack_steps: Vec<AttackStep>,
    pub success_probability: f64,
    pub detection_difficulty: f64,
    pub impact_potential: f64,
    pub mitigation_complexity: f64,
    pub vector_variants: Vec<VectorVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpoofingDetector {
    pub name: String,
    pub detection_patterns: Vec<String>,
    pub trait_specific_checks: HashMap<String, Vec<String>>,
    pub identity_verification_requirements: Vec<String>,
    pub authentication_bypass_patterns: Vec<String>,
    pub spoofing_indicators: Vec<SpoofingIndicator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TamperingDetector {
    pub name: String,
    pub integrity_checks: Vec<IntegrityCheck>,
    pub modification_patterns: Vec<String>,
    pub data_tampering_vectors: Vec<String>,
    pub code_injection_patterns: Vec<String>,
    pub tampering_indicators: Vec<TamperingIndicator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepudiationDetector {
    pub name: String,
    pub audit_trail_requirements: Vec<String>,
    pub non_repudiation_mechanisms: Vec<String>,
    pub logging_patterns: Vec<String>,
    pub evidence_collection_methods: Vec<String>,
    pub repudiation_risks: Vec<RepudiationRisk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationDisclosureDetector {
    pub name: String,
    pub data_leakage_patterns: Vec<String>,
    pub privacy_violations: Vec<String>,
    pub information_exposure_vectors: Vec<String>,
    pub data_classification_requirements: Vec<String>,
    pub disclosure_indicators: Vec<DisclosureIndicator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenialOfServiceDetector {
    pub name: String,
    pub resource_exhaustion_patterns: Vec<String>,
    pub availability_requirements: Vec<String>,
    pub dos_vectors: Vec<String>,
    pub rate_limiting_requirements: Vec<String>,
    pub dos_indicators: Vec<DosIndicator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElevationOfPrivilegeDetector {
    pub name: String,
    pub privilege_escalation_patterns: Vec<String>,
    pub access_control_requirements: Vec<String>,
    pub authorization_bypass_patterns: Vec<String>,
    pub privilege_boundaries: Vec<String>,
    pub escalation_indicators: Vec<EscalationIndicator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackPattern {
    pub pattern_id: String,
    pub name: String,
    pub description: String,
    pub attack_phases: Vec<AttackPhase>,
    pub required_capabilities: Vec<String>,
    pub indicators: Vec<String>,
    pub countermeasures: Vec<String>,
    pub pattern_variations: Vec<PatternVariation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTreeTemplate {
    pub template_id: String,
    pub name: String,
    pub root_goal: String,
    pub node_templates: Vec<NodeTemplate>,
    pub connection_rules: Vec<ConnectionRule>,
    pub template_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatActor {
    pub actor_id: String,
    pub name: String,
    pub actor_type: ThreatActorType,
    pub motivation: Vec<String>,
    pub capabilities: ThreatCapabilities,
    pub resources: ThreatResources,
    pub target_preferences: Vec<String>,
    pub attack_patterns: Vec<String>,
    pub geographic_focus: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatActorType {
    NationState,
    CriminalOrganization,
    Hacktivist,
    InsiderThreat,
    ScriptKiddie,
    Competitor,
    Terrorist,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatCapabilities {
    pub technical_sophistication: f64,
    pub resource_availability: f64,
    pub stealth_capability: f64,
    pub persistence_capability: f64,
    pub social_engineering_skills: f64,
    pub zero_day_access: bool,
    pub insider_access: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatTimeline {
    pub reconnaissance_phase: Duration,
    pub initial_access_phase: Duration,
    pub persistence_phase: Duration,
    pub privilege_escalation_phase: Duration,
    pub lateral_movement_phase: Duration,
    pub data_collection_phase: Duration,
    pub exfiltration_phase: Duration,
    pub cleanup_phase: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatModelingResult {
    pub model_id: String,
    pub analysis_timestamp: SystemTime,
    pub stride_analysis: StrideAnalysisResult,
    pub attack_trees: Vec<AttackTree>,
    pub threat_scenarios: Vec<ThreatScenario>,
    pub attack_vectors: Vec<AttackVector>,
    pub threat_landscape: ThreatLandscapeAssessment,
    pub intelligence_insights: Vec<IntelligenceInsight>,
    pub risk_prioritization: Vec<ThreatRiskPriority>,
    pub mitigation_recommendations: Vec<MitigationRecommendation>,
    pub model_confidence: f64,
    pub model_metadata: HashMap<String, String>,
    pub identified_threats: Vec<IdentifiedThreat>,
    pub overall_risk_score: f64,
}

/// Convenience alias for [`ThreatModelingResult`], used by the outer crate's public API
/// surface (the security analysis framework refers to threat modeling output uniformly as
/// a "threat analysis result" alongside `CryptographicAnalysisResult`, `ComplianceStatus`,
/// etc.).
pub type ThreatAnalysisResult = ThreatModelingResult;

/// A single, concrete security threat identified during comprehensive threat modeling,
/// derived from the STRIDE analysis and attack vector assessment performed by
/// [`super::ThreatModelingEngine::analyze_threats`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifiedThreat {
    pub id: String,
    pub name: String,
    pub severity: ThreatSeverity,
    pub mitigation_strategy: String,
    pub mitigation_complexity: ImplementationEffort,
    pub mitigation_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrideAnalysisResult {
    pub analysis_id: String,
    pub spoofing_threats: Vec<SpoofingThreat>,
    pub tampering_threats: Vec<TamperingThreat>,
    pub repudiation_threats: Vec<RepudiationThreat>,
    pub information_disclosure_threats: Vec<InformationDisclosureThreat>,
    pub denial_of_service_threats: Vec<DenialOfServiceThreat>,
    pub elevation_of_privilege_threats: Vec<ElevationOfPrivilegeThreat>,
    pub composite_threats: Vec<CompositeThreat>,
    pub stride_scores: HashMap<StrideCategory, f64>,
    pub overall_stride_rating: f64,
    pub confidence_intervals: HashMap<StrideCategory, (f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTree {
    pub tree_id: String,
    pub root_node: AttackNode,
    pub attack_paths: Vec<AttackPath>,
    pub critical_paths: Vec<CriticalPath>,
    pub success_probability: f64,
    pub attack_cost: f64,
    pub detection_probability: f64,
    pub tree_metrics: AttackTreeMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackNode {
    pub node_id: String,
    pub node_type: AttackNodeType,
    pub description: String,
    pub children: Vec<AttackNode>,
    pub gate_type: Option<LogicGate>,
    pub success_probability: f64,
    pub attack_cost: f64,
    pub skill_required: f64,
    pub detection_probability: f64,
    pub impact_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttackNodeType {
    Goal,
    Subgoal,
    Action,
    Condition,
    Defense,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicGate {
    And,
    Or,
    Not,
    Xor,
    KOfN(usize, usize),
}

impl SpoofingDetector {
    /// Evaluate identity-verification-related context flags, producing zero or more
    /// spoofing threats.
    pub fn detect_spoofing_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<SpoofingThreat>, ThreatModelingError> {
        let mut threats = Vec::new();
        if context.requires_elevated_privileges && !context.has_access_controls {
            threats.push(SpoofingThreat {
                threat_id: format!("{}-priv", self.name),
                description: "Elevated operations lack identity verification".to_string(),
                severity: ThreatSeverity::High,
                affected_component: context.trait_name.clone(),
                confidence: 0.7,
            });
        }
        if context.has_user_input && !context.has_input_validation {
            threats.push(SpoofingThreat {
                threat_id: format!("{}-input", self.name),
                description: "Unvalidated input may enable identity spoofing".to_string(),
                severity: ThreatSeverity::Medium,
                affected_component: context.trait_name.clone(),
                confidence: 0.5,
            });
        }
        Ok(threats)
    }
}

impl TamperingDetector {
    /// Evaluate integrity-related context flags, producing zero or more tampering threats.
    pub fn detect_tampering_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<TamperingThreat>, ThreatModelingError> {
        let mut threats = Vec::new();
        if context.has_unsafe_operations && !context.has_bounds_checking {
            threats.push(TamperingThreat {
                threat_id: format!("{}-mem", self.name),
                description: "Unsafe memory access without bounds checking enables tampering"
                    .to_string(),
                severity: ThreatSeverity::High,
                affected_component: context.trait_name.clone(),
                confidence: 0.75,
            });
        }
        if context.has_sql_operations && !context.has_parameterized_queries {
            threats.push(TamperingThreat {
                threat_id: format!("{}-sql", self.name),
                description: "Non-parameterized SQL allows data tampering".to_string(),
                severity: ThreatSeverity::Critical,
                affected_component: context.trait_name.clone(),
                confidence: 0.8,
            });
        }
        Ok(threats)
    }
}

impl RepudiationDetector {
    /// Evaluate audit-trail-related context flags, producing zero or more repudiation
    /// threats.
    pub fn detect_repudiation_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<RepudiationThreat>, ThreatModelingError> {
        let mut threats = Vec::new();
        if !context.has_audit_logging {
            threats.push(RepudiationThreat {
                threat_id: format!("{}-audit", self.name),
                description: "Absence of audit logging allows actions to be repudiated".to_string(),
                severity: ThreatSeverity::Medium,
                affected_component: context.trait_name.clone(),
                confidence: 0.6,
            });
        }
        Ok(threats)
    }
}

impl InformationDisclosureDetector {
    /// Evaluate data-protection-related context flags, producing zero or more information
    /// disclosure threats.
    pub fn detect_disclosure_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<InformationDisclosureThreat>, ThreatModelingError> {
        let mut threats = Vec::new();
        if context.handles_sensitive_data && !context.has_encryption {
            threats.push(InformationDisclosureThreat {
                threat_id: format!("{}-plaintext", self.name),
                description: "Sensitive data is handled without encryption".to_string(),
                severity: ThreatSeverity::High,
                affected_component: context.trait_name.clone(),
                confidence: 0.8,
            });
        }
        if context.handles_personal_data && !context.has_data_anonymization {
            threats.push(InformationDisclosureThreat {
                threat_id: format!("{}-pii", self.name),
                description: "Personal data is processed without anonymization".to_string(),
                severity: ThreatSeverity::Critical,
                affected_component: context.trait_name.clone(),
                confidence: 0.75,
            });
        }
        Ok(threats)
    }
}

impl DenialOfServiceDetector {
    /// Evaluate resource-management-related context flags, producing zero or more denial
    /// of service threats.
    pub fn detect_dos_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<DenialOfServiceThreat>, ThreatModelingError> {
        let mut threats = Vec::new();
        if context.has_resource_intensive_operations && !context.has_rate_limiting {
            threats.push(DenialOfServiceThreat {
                threat_id: format!("{}-resource", self.name),
                description: "Resource-intensive operations lack rate limiting".to_string(),
                severity: ThreatSeverity::Medium,
                affected_component: context.trait_name.clone(),
                confidence: 0.65,
            });
        }
        if context.has_unbounded_recursion {
            threats.push(DenialOfServiceThreat {
                threat_id: format!("{}-recursion", self.name),
                description: "Unbounded recursion can exhaust the call stack".to_string(),
                severity: ThreatSeverity::High,
                affected_component: context.trait_name.clone(),
                confidence: 0.7,
            });
        }
        Ok(threats)
    }
}

impl ElevationOfPrivilegeDetector {
    /// Evaluate privilege-boundary-related context flags, producing zero or more elevation
    /// of privilege threats.
    pub fn detect_escalation_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ElevationOfPrivilegeThreat>, ThreatModelingError> {
        let mut threats = Vec::new();
        if context.requires_elevated_privileges && !context.has_privilege_separation {
            threats.push(ElevationOfPrivilegeThreat {
                threat_id: format!("{}-separation", self.name),
                description: "Elevated privileges are not separated from normal operation"
                    .to_string(),
                severity: ThreatSeverity::Critical,
                affected_component: context.trait_name.clone(),
                confidence: 0.8,
            });
        }
        if context.has_dynamic_dispatch && !context.has_type_safety_checks {
            threats.push(ElevationOfPrivilegeThreat {
                threat_id: format!("{}-dispatch", self.name),
                description:
                    "Dynamic dispatch without type-safety checks may enable privilege confusion"
                        .to_string(),
                severity: ThreatSeverity::Medium,
                affected_component: context.trait_name.clone(),
                confidence: 0.5,
            });
        }
        Ok(threats)
    }
}

impl ThreatIntelligenceSource {
    /// Produce zero or more intelligence insights relevant to this source's focus, given
    /// the trait usage context under analysis.
    pub fn gather_insights(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<IntelligenceInsight>, ThreatModelingError> {
        let mut insights = Vec::new();
        if context.has_cryptographic_operations {
            insights.push(IntelligenceInsight {
                insight_id: format!("{}-crypto", self.source_id),
                description: format!(
                    "{} reports active targeting of cryptographic implementations",
                    self.name
                ),
                confidence: self.reliability,
                relevance: 0.7,
                source_ids: vec![self.source_id.clone()],
            });
        }
        if context.requires_elevated_privileges {
            insights.push(IntelligenceInsight {
                insight_id: format!("{}-priv", self.source_id),
                description: format!(
                    "{} tracks active campaigns against privileged interfaces",
                    self.name
                ),
                confidence: self.reliability * 0.9,
                relevance: 0.6,
                source_ids: vec![self.source_id.clone()],
            });
        }
        Ok(insights)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatModelingError {
    AnalysisError(String),
    DataError(String),
    ModelingError(String),
    IntelligenceError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for ThreatModelingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreatModelingError::AnalysisError(msg) => write!(f, "Analysis error: {}", msg),
            ThreatModelingError::DataError(msg) => write!(f, "Data error: {}", msg),
            ThreatModelingError::ModelingError(msg) => write!(f, "Modeling error: {}", msg),
            ThreatModelingError::IntelligenceError(msg) => write!(f, "Intelligence error: {}", msg),
            ThreatModelingError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ThreatModelingError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatModelingConfig {
    pub stride_weights: HashMap<StrideCategory, f64>,
    pub intelligence_sources: Vec<String>,
    pub model_confidence_threshold: f64,
    pub cache_duration: Duration,
    pub analysis_depth: AnalysisDepth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    Surface,
    Moderate,
    Deep,
    Comprehensive,
}

impl Default for ThreatModelingConfig {
    fn default() -> Self {
        let mut stride_weights = HashMap::new();
        stride_weights.insert(StrideCategory::Spoofing, 1.0);
        stride_weights.insert(StrideCategory::Tampering, 1.0);
        stride_weights.insert(StrideCategory::Repudiation, 1.0);
        stride_weights.insert(StrideCategory::InformationDisclosure, 1.0);
        stride_weights.insert(StrideCategory::DenialOfService, 1.0);
        stride_weights.insert(StrideCategory::ElevationOfPrivilege, 1.0);

        Self {
            stride_weights,
            intelligence_sources: vec!["mitre_att&ck".to_string(), "nist_nvd".to_string()],
            model_confidence_threshold: 0.8,
            cache_duration: Duration::from_secs(3600),
            analysis_depth: AnalysisDepth::Moderate,
        }
    }
}

macro_rules! define_supporting_types {
    () => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct SpoofingIndicator {
            pub indicator_type: String,
            pub pattern: String,
            pub confidence: f64,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct TamperingIndicator {
            pub indicator_type: String,
            pub modification_type: String,
            pub detection_method: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct IntegrityCheck {
            pub check_type: String,
            pub algorithm: String,
            pub scope: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct RepudiationRisk {
            pub risk_type: String,
            pub likelihood: f64,
            pub impact: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct DisclosureIndicator {
            pub data_type: String,
            pub exposure_method: String,
            pub sensitivity_level: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct DosIndicator {
            pub attack_type: String,
            pub resource_target: String,
            pub impact_level: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct EscalationIndicator {
            pub escalation_type: String,
            pub target_privilege: String,
            pub method: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct AttackPhase {
            pub phase_name: String,
            pub description: String,
            pub techniques: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct PatternVariation {
            pub variation_name: String,
            pub differences: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct NodeTemplate {
            pub template_id: String,
            pub node_type: AttackNodeType,
            pub description_template: String,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ConnectionRule {
            pub from_template: String,
            pub to_template: String,
            pub gate_type: LogicGate,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ThreatResources {
            pub financial_resources: f64,
            pub technical_resources: f64,
            pub human_resources: f64,
            pub time_resources: f64,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ImpactAssessment {
            pub confidentiality_impact: f64,
            pub integrity_impact: f64,
            pub availability_impact: f64,
            pub financial_impact: f64,
            pub reputational_impact: f64,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct DetectionMethod {
            pub method_name: String,
            pub detection_probability: f64,
            pub false_positive_rate: f64,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct MitigationStrategy {
            pub strategy_name: String,
            pub effectiveness: f64,
            pub implementation_cost: f64,
            pub implementation_time: Duration,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ScenarioVariant {
            pub variant_id: String,
            pub description: String,
            pub probability_modifier: f64,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct AttackSurface {
            pub network_surface: Vec<String>,
            pub application_surface: Vec<String>,
            pub physical_surface: Vec<String>,
            pub human_surface: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct EntryPoint {
            pub entry_id: String,
            pub entry_type: String,
            pub accessibility: f64,
            pub security_controls: Vec<String>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct AttackStep {
            pub step_id: String,
            pub description: String,
            pub required_skills: Vec<String>,
            pub success_probability: f64,
            pub detection_probability: f64,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct VectorVariant {
            pub variant_id: String,
            pub modifications: Vec<String>,
            pub effectiveness_change: f64,
        }
    };
}

define_supporting_types!();

// ------------------------------------------------------------------------------------------
// Additional threat modeling support types.
//
// These are intentionally simple, flat data containers (2-6 fields of String/f64/bool/Vec),
// mirroring the leaf types generated by `define_supporting_types!()` above and the pattern
// used throughout `core_analyzer.rs`. The `*Manager`/`*Analyzer`/`*Generator` "container"
// types among them are largely inert configuration/storage; the actual heuristics that read
// `TraitUsageContext` live on `ThreatModelingEngine` and the six STRIDE `*Detector` types.
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpoofingThreat {
    pub threat_id: String,
    pub description: String,
    pub severity: ThreatSeverity,
    pub affected_component: String,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TamperingThreat {
    pub threat_id: String,
    pub description: String,
    pub severity: ThreatSeverity,
    pub affected_component: String,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepudiationThreat {
    pub threat_id: String,
    pub description: String,
    pub severity: ThreatSeverity,
    pub affected_component: String,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationDisclosureThreat {
    pub threat_id: String,
    pub description: String,
    pub severity: ThreatSeverity,
    pub affected_component: String,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenialOfServiceThreat {
    pub threat_id: String,
    pub description: String,
    pub severity: ThreatSeverity,
    pub affected_component: String,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElevationOfPrivilegeThreat {
    pub threat_id: String,
    pub description: String,
    pub severity: ThreatSeverity,
    pub affected_component: String,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeThreat {
    pub composite_id: String,
    pub contributing_categories: Vec<StrideCategory>,
    pub description: String,
    pub combined_severity: ThreatSeverity,
    pub amplification_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualStrideAnalyzer {
    pub context_key: String,
    pub adjustment_factor: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackNodeGenerator {
    pub generator_id: String,
    pub node_type: AttackNodeType,
    pub generation_rules: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTreeOptimization {
    pub optimization_strategy: String,
    pub pruning_threshold: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityCalculator {
    pub calculator_id: String,
    pub base_probability: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBenefitAnalyzer {
    pub analyzer_id: String,
    pub cost_weight: f64,
    pub benefit_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIntelligenceSource {
    pub source_id: String,
    pub name: String,
    pub reliability: f64,
    pub source_type: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatFeed {
    pub feed_id: String,
    pub provider: String,
    pub update_frequency: Duration,
    pub entry_count: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorOfCompromise {
    pub ioc_id: String,
    pub indicator_type: String,
    pub value: String,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackCampaign {
    pub campaign_id: String,
    pub name: String,
    pub associated_actors: Vec<String>,
    pub active: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatActorProfile {
    pub actor_id: String,
    pub sophistication: f64,
    pub known_ttps: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceCorrelation {
    pub correlation_method: String,
    pub min_confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedAggregator {
    pub aggregation_strategy: String,
    pub source_count: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceScoring {
    pub scoring_model: String,
    pub min_confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEnvironment {
    pub environment_type: String,
    pub threat_density: f64,
    pub maturity_level: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergingThreat {
    pub threat_name: String,
    pub description: String,
    pub growth_rate: f64,
    pub first_observed: SystemTime,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatTrend {
    pub trend_name: String,
    pub direction: TrendDirection,
    pub magnitude: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicThreatFactor {
    pub region: String,
    pub risk_multiplier: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndustryThreat {
    pub industry: String,
    pub threat_description: String,
    pub prevalence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnologyThreat {
    pub technology: String,
    pub vulnerability_class: String,
    pub exposure: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEvolutionModel {
    pub model_name: String,
    pub projected_growth: f64,
    pub time_horizon: Duration,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandscapeMetrics {
    pub total_threats_tracked: u32,
    pub average_severity: f64,
    pub trend_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceInsight {
    pub insight_id: String,
    pub description: String,
    pub confidence: f64,
    pub relevance: f64,
    pub source_ids: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatRiskPriority {
    pub threat_category: String,
    pub risk_score: f64,
    pub priority_level: MitigationPriority,
    pub justification: String,
    pub recommended_timeline: Duration,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationRecommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub priority: MitigationPriority,
    pub estimated_effort: ImplementationEffort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackPath {
    pub path_id: String,
    pub steps: Vec<AttackStep>,
    pub success_probability: f64,
    pub total_cost: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    pub path_id: String,
    pub risk_score: f64,
    pub steps: Vec<AttackStep>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTreeMetrics {
    pub total_nodes: usize,
    pub max_depth: usize,
    pub average_branching_factor: f64,
    pub complexity_score: f64,
}

impl AttackTreeOptimization {
    pub fn new() -> Self {
        Self {
            optimization_strategy: "prune_low_probability".to_string(),
            pruning_threshold: 0.05,
        }
    }
}
impl Default for AttackTreeOptimization {
    fn default() -> Self {
        Self::new()
    }
}

impl IntelligenceCorrelation {
    pub fn new() -> Self {
        Self {
            correlation_method: "confidence_weighted".to_string(),
            min_confidence: 0.5,
        }
    }
}
impl Default for IntelligenceCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedAggregator {
    pub fn new() -> Self {
        Self {
            aggregation_strategy: "union".to_string(),
            source_count: 0,
        }
    }
}
impl Default for FeedAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl IntelligenceScoring {
    pub fn new() -> Self {
        Self {
            scoring_model: "weighted_average".to_string(),
            min_confidence_threshold: 0.5,
        }
    }
}
impl Default for IntelligenceScoring {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreatEnvironment {
    pub fn new() -> Self {
        Self {
            environment_type: "unknown".to_string(),
            threat_density: 0.0,
            maturity_level: "unassessed".to_string(),
        }
    }
}
impl Default for ThreatEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl LandscapeMetrics {
    pub fn new() -> Self {
        Self {
            total_threats_tracked: 0,
            average_severity: 0.0,
            trend_score: 0.0,
        }
    }
}
impl Default for LandscapeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedThreatModel {
    pub result: ThreatModelingResult,
    pub cache_timestamp: SystemTime,
    pub cache_ttl: Duration,
}
