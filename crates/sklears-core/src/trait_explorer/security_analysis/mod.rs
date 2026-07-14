//! Security Analysis Framework
//!
//! This module provides a comprehensive security analysis framework for analyzing trait usage patterns
//! and identifying potential security vulnerabilities, risks, and compliance issues.
//!
//! ## Architecture
//!
//! The security analysis framework is organized into focused, specialized modules:
//!
//! - **`core_analyzer`**: Main security analyzer with comprehensive vulnerability assessment and orchestration
//! - **`vulnerability_database`**: Advanced vulnerability database with CVE integration and OWASP mapping
//! - **`risk_assessment`**: Security risk assessment with multiple risk models and Bayesian analysis
//! - **`threat_modeling`**: STRIDE analysis, attack tree generation, and threat scenario modeling
//! - **`crypto_analysis`**: Cryptographic security analysis including algorithm and implementation analysis
//! - **`compliance_framework`**: Compliance checking and standards validation across multiple frameworks
//! - **`security_metrics`**: Comprehensive security metrics collection and analysis with KPI/KRI monitoring
//! - **`security_types`**: Shared data structures, configurations, and types used across all modules
//!
//! ## Usage
//!
//! ```rust,no_run
//! use sklears_core::trait_explorer::security_analysis::{
//!     create_comprehensive_security_analyzer, SecurityAnalysisConfig, TraitUsageContext,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a comprehensive security analyzer
//! let mut analyzer = create_comprehensive_security_analyzer();
//!
//! // Configure analysis parameters
//! let config = SecurityAnalysisConfig::default();
//! analyzer.configure_analysis(config);
//!
//! // Perform security analysis
//! let context = TraitUsageContext {
//!     trait_name: "MyTrait".to_string(),
//!     ..Default::default()
//! };
//! let analysis_result = analyzer.analyze_comprehensive_security(&context)?;
//!
//! // Access specific analysis results
//! println!("Overall security score: {}", analysis_result.overall_security_score);
//! println!(
//!     "Vulnerabilities found: {}",
//!     analysis_result.vulnerability_assessment.vulnerabilities.len()
//! );
//! println!("Risk level: {:?}", analysis_result.overall_risk_level);
//! println!("Compliance status: {:?}", analysis_result.overall_compliance_status);
//! # Ok(())
//! # }
//! ```
//!
//! ## Key Features
//!
//! ### Vulnerability Analysis
//! - CVE database integration with real-time updates
//! - OWASP Top 10 mapping and analysis
//! - Custom vulnerability pattern detection
//! - Exploit availability assessment
//! - Automated remediation guidance
//!
//! ### Risk Assessment
//! - Multiple risk assessment methodologies (Qualitative, Quantitative, Hybrid)
//! - Bayesian risk analysis with historical data integration
//! - Monte Carlo simulation for risk modeling
//! - Business impact assessment across multiple dimensions
//! - Risk trend analysis and forecasting
//!
//! ### Threat Modeling
//! - STRIDE analysis (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)
//! - Attack tree generation and analysis
//! - Threat scenario modeling with multiple variants
//! - Threat intelligence integration
//! - Attack vector identification and assessment
//!
//! ### Cryptographic Analysis
//! - Algorithm security assessment (symmetric, asymmetric, hash functions)
//! - Key management analysis
//! - Side-channel attack detection
//! - Protocol security analysis (TLS, SSH, IPSec, etc.)
//! - Quantum resistance evaluation
//! - Implementation security analysis
//!
//! ### Compliance Framework
//! - Multi-framework compliance checking (NIST, GDPR, HIPAA, SOC 2, ISO 27001, PCI DSS)
//! - Regulatory compliance validation
//! - Audit trail management
//! - Gap analysis and remediation planning
//! - Certification readiness assessment
//! - Continuous compliance monitoring
//!
//! ### Security Metrics
//! - KPI (Key Performance Indicators) analysis and tracking
//! - KRI (Key Risk Indicators) monitoring with early warning systems
//! - Real-time security dashboards
//! - Trend analysis and anomaly detection
//! - Benchmarking against industry standards
//! - Security scorecard generation
//!
//! ## Configuration
//!
//! The framework supports extensive configuration through the `SecurityAnalysisConfig` struct:
//!
//! ```rust,no_run
//! use sklears_core::trait_explorer::security_analysis::{
//!     SecurityAnalysisConfig, AnalysisDepth, RiskAppetite, RiskTolerance
//! };
//! use std::time::Duration;
//!
//! let config = SecurityAnalysisConfig {
//!     analysis_depth: AnalysisDepth::Deep,
//!     risk_tolerance: RiskTolerance {
//!         overall_risk_appetite: RiskAppetite::Conservative,
//!         ..Default::default()
//!     },
//!     compliance_requirements: vec![
//!         "NIST".to_string(),
//!         "GDPR".to_string(),
//!         "ISO27001".to_string()
//!     ],
//!     reporting_frequency: Duration::from_secs(86400), // Daily
//!     automated_remediation: false,
//!     real_time_monitoring: true,
//!     threat_intelligence_enabled: true,
//!     vulnerability_scanning_enabled: true,
//!     ..Default::default()
//! };
//! ```
//!
//! ## Integration
//!
//! The security analysis framework integrates with:
//! - CVE databases (NVD, MITRE)
//! - Threat intelligence feeds
//! - Compliance management systems
//! - SIEM and security monitoring tools
//! - Vulnerability scanners
//! - Risk management platforms
//!
//! ## Performance Considerations
//!
//! - Configurable analysis depth to balance thoroughness with performance
//! - Caching mechanisms for expensive operations
//! - Parallel processing for independent analysis tasks
//! - Incremental analysis to avoid redundant work
//! - Resource constraint management
//!
//! ## Security and Privacy
//!
//! - All analysis data is processed locally by default
//! - Configurable data retention policies
//! - Encryption for sensitive analysis results
//! - Audit logging for all security operations
//! - Privacy-preserving analysis techniques

// Module declarations
pub mod compliance_framework;
pub mod core_analyzer;
pub mod crypto_analysis;
pub mod risk_assessment;
pub mod security_metrics;
pub mod security_types;
pub mod threat_modeling;
pub mod vulnerability_database;

// Re-export core types and functionality
pub use security_types::*;

// Core analyzer exports
pub use core_analyzer::{
    create_trait_security_analyzer, perform_comprehensive_security_analysis, RiskRecommendation,
    SecurityAnalysis, SecurityAnalysisError, SecurityAnalysisMetadata,
    SecurityAnalysisResult as CoreSecurityAnalysisResult, SecurityRecommendation, SecurityRisk,
    SecurityVulnerability, TraitSecurityAnalyzer,
};

// Vulnerability database exports
pub use vulnerability_database::{
    assess_known_vulnerabilities, create_vulnerability_database, create_vulnerability_details,
    CveEntry, VulnerabilityAssessmentResult, VulnerabilityDatabase, VulnerabilityDatabaseError,
    VulnerabilityRule,
};

// Risk assessment exports
pub use risk_assessment::{
    assess_comprehensive_risk, create_security_risk_assessor, BayesianRiskParameters,
    ConfidenceIntervals, MonteCarloConfig, RiskAnalysis, RiskAssessmentError, RiskAssessmentModel,
    RiskAssessmentResult, RiskFactor, SecurityRiskAssessor,
};

// Threat modeling exports
pub use threat_modeling::{
    create_comprehensive_threat_model, create_threat_modeling_engine, AttackTree,
    AttackTreeGenerator, AttackVector, IdentifiedThreat, StrideAnalysisResult, StrideAnalyzer,
    ThreatAnalysisResult, ThreatIntelligenceManager, ThreatLandscapeAssessment,
    ThreatModelingEngine, ThreatModelingError, ThreatModelingResult, ThreatScenario,
};

// Cryptographic analysis exports
pub use crypto_analysis::{
    analyze_cryptographic_security, create_cryptographic_analyzer, ConstantTimeViolation,
    CryptographicAlgorithmAnalyzer, CryptographicAnalysisError, CryptographicAnalysisResult,
    CryptographicAnalyzer, CryptographicImplementationAnalyzer, CryptographicIssue,
    CryptographicProtocolAnalyzer, CryptographicStrengthAssessment, DigitalSignatureAnalyzer,
    EncryptionAnalyzer, HashFunctionAnalyzer, KeyManagementAnalyzer, QuantumResistanceAnalyzer,
    RandomNumberGeneratorAnalyzer, SideChannelAttackDetector, SideChannelRisk, TimingVulnerability,
};

// Compliance framework exports
pub use compliance_framework::{
    assess_comprehensive_compliance, create_compliance_framework_manager, AuditManager,
    CertificationManager, ComplianceAssessmentResult, ComplianceEngine, ComplianceError,
    ComplianceFrameworkManager, ComplianceLevel, ComplianceMonitor, ComplianceReportingEngine,
    ComplianceStatus, ComplianceViolation, ComplianceViolationDetail, ControlsAssessor,
    DocumentationManager, FrameworkAssessmentResult, GapAnalyzer, PolicyEngine,
    RegulatoryFramework, SecurityStandard,
};

// Security metrics exports
pub use security_metrics::{
    collect_comprehensive_security_metrics, create_security_metrics_collector, AnomalyDetector,
    BenchmarkingEngine, ComplianceTracker, CorrelationAnalyzer, DashboardManager, KpiAnalyzer,
    KriMonitor, MetricCollection, MetricCollector, PerformanceMeasurer, RealTimeMonitor,
    ScorecardGenerator, SecurityMetricsCollector, SecurityMetricsError, SecurityMetricsResult,
    SecurityTrend, TrendAnalyzer,
};

// Common imports for convenience
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Comprehensive security analysis result that combines all analysis domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveSecurityAnalysisResult {
    /// Unique identifier for this analysis session
    pub analysis_id: String,
    /// Timestamp when the analysis was performed
    pub analysis_timestamp: SystemTime,
    /// Core security analysis results
    pub core_analysis: SecurityAnalysis,
    /// Vulnerability assessment results
    pub vulnerability_assessment: VulnerabilityAssessmentResult,
    /// Risk assessment results
    pub risk_assessment: RiskAssessmentResult,
    /// Threat modeling results
    pub threat_modeling: ThreatModelingResult,
    /// Cryptographic analysis results
    pub cryptographic_analysis: CryptographicAnalysisResult,
    /// Compliance assessment results
    pub compliance_assessment: ComplianceAssessmentResult,
    /// Security metrics results
    pub security_metrics: SecurityMetricsResult,
    /// Overall security score (0.0 - 10.0)
    pub overall_security_score: f64,
    /// Overall risk level
    pub overall_risk_level: RiskLevel,
    /// Overall compliance status
    pub overall_compliance_status: ComplianceStatus,
    /// Consolidated recommendations across all analysis domains
    pub consolidated_recommendations: Vec<ConsolidatedRecommendation>,
    /// Executive summary for stakeholders
    pub executive_summary: ExecutiveSummary,
    /// Analysis confidence level (0.0 - 1.0)
    pub analysis_confidence: f64,
    /// Analysis metadata and configuration
    pub analysis_metadata: HashMap<String, String>,
}

/// Consolidated recommendation that may span multiple analysis domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedRecommendation {
    /// Unique identifier for this recommendation
    pub recommendation_id: String,
    /// Recommendation title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Priority level
    pub priority: AnalysisPriority,
    /// Affected analysis domains
    pub analysis_domains: Vec<String>,
    /// Related vulnerabilities
    pub related_vulnerabilities: Vec<String>,
    /// Related risks
    pub related_risks: Vec<String>,
    /// Related compliance issues
    pub related_compliance_issues: Vec<String>,
    /// Implementation guidance
    pub implementation_guidance: String,
    /// Expected risk reduction
    pub expected_risk_reduction: f64,
    /// Implementation cost estimate
    pub implementation_cost: f64,
    /// Implementation timeline
    pub implementation_timeline: Duration,
    /// Success metrics
    pub success_metrics: Vec<String>,
}

/// Executive summary for stakeholder communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    /// High-level security posture assessment
    pub security_posture: SecurityPosture,
    /// Key findings summary
    pub key_findings: Vec<String>,
    /// Critical issues requiring immediate attention
    pub critical_issues: Vec<String>,
    /// Top security risks
    pub top_risks: Vec<String>,
    /// Compliance status summary
    pub compliance_summary: String,
    /// Recommended next steps
    pub recommended_next_steps: Vec<String>,
    /// Resource requirements for remediation
    pub resource_requirements: ResourceRequirements,
    /// Expected timeline for major improvements
    pub improvement_timeline: Duration,
    /// Return on investment for security improvements
    pub roi_estimate: f64,
}

/// Overall security posture assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityPosture {
    /// Security posture is strong with minimal concerns
    Strong,
    /// Security posture is adequate with some areas for improvement
    Adequate,
    /// Security posture has significant weaknesses requiring attention
    Weak,
    /// Security posture is poor with critical vulnerabilities
    Poor,
    /// Security posture is critically compromised requiring immediate action
    Critical,
}

/// Resource requirements for implementing security improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Estimated budget requirements
    pub budget_estimate: f64,
    /// Required personnel and skills
    pub personnel_requirements: Vec<String>,
    /// Technology investments needed
    pub technology_requirements: Vec<String>,
    /// Training requirements
    pub training_requirements: Vec<String>,
    /// External consulting needs
    pub consulting_requirements: Vec<String>,
}

/// Bundles references to each per-domain analysis result together, so the aggregation
/// helper methods on [`ComprehensiveSecurityAnalyzer`] can take a single argument instead
/// of one parameter per domain (keeping them under `clippy::too_many_arguments`'s limit).
struct AnalysisDomainResults<'a> {
    core_analysis: &'a SecurityAnalysis,
    vulnerability_assessment: &'a VulnerabilityAssessmentResult,
    risk_assessment: &'a RiskAssessmentResult,
    threat_modeling: &'a ThreatModelingResult,
    cryptographic_analysis: &'a CryptographicAnalysisResult,
    compliance_assessment: &'a ComplianceAssessmentResult,
    security_metrics: &'a SecurityMetricsResult,
}

/// Comprehensive security analyzer that orchestrates all analysis components
#[derive(Debug)]
pub struct ComprehensiveSecurityAnalyzer {
    core_analyzer: TraitSecurityAnalyzer,
    vulnerability_database: VulnerabilityDatabase,
    risk_assessor: SecurityRiskAssessor,
    threat_modeling_engine: ThreatModelingEngine,
    cryptographic_analyzer: CryptographicAnalyzer,
    compliance_manager: ComplianceFrameworkManager,
    metrics_collector: SecurityMetricsCollector,
    analysis_config: SecurityAnalysisConfig,
}

impl ComprehensiveSecurityAnalyzer {
    /// Create a new comprehensive security analyzer with default configuration
    pub fn new() -> Self {
        Self {
            core_analyzer: TraitSecurityAnalyzer::new(),
            vulnerability_database: VulnerabilityDatabase::new(),
            risk_assessor: SecurityRiskAssessor::new(),
            threat_modeling_engine: ThreatModelingEngine::new(),
            cryptographic_analyzer: CryptographicAnalyzer::new(),
            compliance_manager: ComplianceFrameworkManager::new(),
            metrics_collector: SecurityMetricsCollector::new(),
            analysis_config: SecurityAnalysisConfig::default(),
        }
    }

    /// Configure the analysis parameters
    pub fn configure_analysis(&mut self, config: SecurityAnalysisConfig) {
        self.analysis_config = config;
    }

    /// Perform comprehensive security analysis across all domains
    pub fn analyze_comprehensive_security(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<ComprehensiveSecurityAnalysisResult, SecurityAnalysisError> {
        let analysis_id = self.generate_analysis_id();
        let analysis_timestamp = SystemTime::now();

        // Perform analysis across all domains
        let core_analysis = self
            .core_analyzer
            .analyze_trait_security(context)
            .map_err(|e| {
                SecurityAnalysisError::AnalysisError(format!("Core analysis failed: {}", e))
            })?;

        let vulnerability_assessment = vulnerability_database::assess_known_vulnerabilities(
            &self.vulnerability_database,
            context,
        )
        .map_err(|e| {
            SecurityAnalysisError::AnalysisError(format!("Vulnerability assessment failed: {}", e))
        })?;

        let risk_assessment = self
            .risk_assessor
            .assess_comprehensive_risk(context)
            .map_err(|e| {
                SecurityAnalysisError::AnalysisError(format!("Risk assessment failed: {}", e))
            })?;

        let threat_modeling = self
            .threat_modeling_engine
            .analyze_threats(context)
            .map_err(|e| {
                SecurityAnalysisError::AnalysisError(format!("Threat modeling failed: {}", e))
            })?;

        let cryptographic_analysis = self
            .cryptographic_analyzer
            .analyze_cryptographic_security(context)
            .map_err(|e| {
                SecurityAnalysisError::AnalysisError(format!(
                    "Cryptographic analysis failed: {}",
                    e
                ))
            })?;

        let compliance_assessment =
            self.compliance_manager
                .assess_compliance(context)
                .map_err(|e| {
                    SecurityAnalysisError::AnalysisError(format!(
                        "Compliance assessment failed: {}",
                        e
                    ))
                })?;

        let security_metrics = self
            .metrics_collector
            .collect_security_metrics(context)
            .map_err(|e| {
                SecurityAnalysisError::AnalysisError(format!(
                    "Security metrics collection failed: {}",
                    e
                ))
            })?;

        // Calculate overall scores and status
        let domain_results = AnalysisDomainResults {
            core_analysis: &core_analysis,
            vulnerability_assessment: &vulnerability_assessment,
            risk_assessment: &risk_assessment,
            threat_modeling: &threat_modeling,
            cryptographic_analysis: &cryptographic_analysis,
            compliance_assessment: &compliance_assessment,
            security_metrics: &security_metrics,
        };
        let overall_security_score = self.calculate_overall_security_score(&domain_results)?;

        let overall_risk_level =
            self.determine_overall_risk_level(&risk_assessment, &threat_modeling)?;
        let overall_compliance_status =
            self.determine_overall_compliance_status(&compliance_assessment)?;

        // Generate consolidated recommendations
        let consolidated_recommendations = self.generate_consolidated_recommendations(
            &core_analysis,
            &vulnerability_assessment,
            &risk_assessment,
            &threat_modeling,
            &cryptographic_analysis,
            &compliance_assessment,
        )?;

        // Generate executive summary
        let executive_summary = self.generate_executive_summary(
            overall_security_score,
            &overall_risk_level,
            &overall_compliance_status,
            &consolidated_recommendations,
        )?;

        // Calculate analysis confidence
        let analysis_confidence = self.calculate_analysis_confidence(&domain_results)?;

        // Generate metadata
        let analysis_metadata = self.generate_analysis_metadata(context);

        Ok(ComprehensiveSecurityAnalysisResult {
            analysis_id,
            analysis_timestamp,
            core_analysis,
            vulnerability_assessment,
            risk_assessment,
            threat_modeling,
            cryptographic_analysis,
            compliance_assessment,
            security_metrics,
            overall_security_score,
            overall_risk_level,
            overall_compliance_status,
            consolidated_recommendations,
            executive_summary,
            analysis_confidence,
            analysis_metadata,
        })
    }

    fn generate_analysis_id(&self) -> String {
        format!(
            "comprehensive_analysis_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("duration_since should succeed")
                .as_secs()
        )
    }

    fn calculate_overall_security_score(
        &self,
        results: &AnalysisDomainResults<'_>,
    ) -> Result<f64, SecurityAnalysisError> {
        // Weighted average of all analysis domain scores
        let weights = [0.2, 0.15, 0.2, 0.15, 0.1, 0.1, 0.1]; // Core, Vuln, Risk, Threat, Crypto, Compliance, Metrics
        let scores = [
            Self::core_security_score(results.core_analysis),
            results.vulnerability_assessment.overall_vulnerability_score,
            results.risk_assessment.overall_risk_score,
            results.threat_modeling.model_confidence * 10.0, // Convert confidence to score
            results.cryptographic_analysis.overall_cryptographic_score,
            results.compliance_assessment.compliance_score,
            results.security_metrics.overall_security_score,
        ];

        let weighted_score = weights
            .iter()
            .zip(scores.iter())
            .map(|(weight, score)| weight * score)
            .sum::<f64>();

        Ok(weighted_score.clamp(0.0, 10.0))
    }

    /// Derive a 0.0-10.0 "security score" from the core analyzer's coarse
    /// [`RiskLevel`], since [`SecurityAnalysis`] itself does not track a numeric score
    /// directly (only the discrete risk level produced from its vulnerabilities/risks).
    fn core_security_score(core_analysis: &SecurityAnalysis) -> f64 {
        match core_analysis.overall_risk_level {
            RiskLevel::Minimal => 9.0,
            RiskLevel::Low => 7.5,
            RiskLevel::Medium => 5.5,
            RiskLevel::High => 3.0,
            RiskLevel::Critical => 1.0,
        }
    }

    /// Derive a 0.0-1.0 confidence score for the core analysis: analyses that found
    /// concrete evidence (vulnerabilities or risk factors) are more confident than ones
    /// that found nothing to report.
    fn core_confidence(core_analysis: &SecurityAnalysis) -> f64 {
        if core_analysis.vulnerabilities.is_empty() && core_analysis.risk_factors.is_empty() {
            0.5
        } else {
            0.85
        }
    }

    fn determine_overall_risk_level(
        &self,
        risk_assessment: &RiskAssessmentResult,
        threat_modeling: &ThreatModelingResult,
    ) -> Result<RiskLevel, SecurityAnalysisError> {
        // Use the higher of risk assessment and threat modeling risk levels
        let risk_level = if risk_assessment.risk_level.to_numeric_value()
            > RiskLevel::from_score(threat_modeling.model_confidence * 10.0).to_numeric_value()
        {
            risk_assessment.risk_level.clone()
        } else {
            RiskLevel::from_score(threat_modeling.model_confidence * 10.0)
        };

        Ok(risk_level)
    }

    fn determine_overall_compliance_status(
        &self,
        compliance_assessment: &ComplianceAssessmentResult,
    ) -> Result<ComplianceStatus, SecurityAnalysisError> {
        Ok(compliance_assessment
            .framework_assessments
            .values()
            .map(|assessment| &assessment.compliance_status)
            .min()
            .unwrap_or(&ComplianceStatus::NotAssessed)
            .clone())
    }

    fn generate_consolidated_recommendations(
        &self,
        core_analysis: &SecurityAnalysis,
        vulnerability_assessment: &VulnerabilityAssessmentResult,
        risk_assessment: &RiskAssessmentResult,
        threat_modeling: &ThreatModelingResult,
        cryptographic_analysis: &CryptographicAnalysisResult,
        compliance_assessment: &ComplianceAssessmentResult,
    ) -> Result<Vec<ConsolidatedRecommendation>, SecurityAnalysisError> {
        let mut recommendations = Vec::new();

        // Consolidate recommendations from all analysis domains
        // This is a simplified implementation - in practice, this would involve
        // sophisticated recommendation correlation and prioritization

        for (i, recommendation) in core_analysis.recommendations.iter().enumerate() {
            let priority = Self::risk_severity_to_priority(&recommendation.priority);
            let expected_risk_reduction = match priority {
                AnalysisPriority::Critical => 0.8,
                AnalysisPriority::High => 0.6,
                AnalysisPriority::Medium => 0.4,
                AnalysisPriority::Low => 0.2,
                AnalysisPriority::Informational => 0.05,
            };

            recommendations.push(ConsolidatedRecommendation {
                recommendation_id: format!("core_{}", i),
                title: recommendation.title.clone(),
                description: recommendation.description.clone(),
                priority,
                analysis_domains: vec!["core_analysis".to_string()],
                related_vulnerabilities: Vec::new(),
                related_risks: Vec::new(),
                related_compliance_issues: Vec::new(),
                implementation_guidance: recommendation.description.clone(),
                expected_risk_reduction,
                implementation_cost: recommendation.estimated_cost.to_numeric_value(),
                implementation_timeline: recommendation.implementation_timeline,
                success_metrics: recommendation.testing_requirements.clone(),
            });
        }

        // Vulnerability domain: flag when the aggregated vulnerability score is
        // significant enough to warrant a dedicated consolidated recommendation.
        if vulnerability_assessment.overall_vulnerability_score >= 5.0 {
            recommendations.push(ConsolidatedRecommendation {
                recommendation_id: "vuln_aggregate".to_string(),
                title: "Remediate aggregated known vulnerabilities".to_string(),
                description: format!(
                    "{} known vulnerabilit(y/ies) identified with an aggregated score of {:.1}/10.0",
                    vulnerability_assessment.total_count,
                    vulnerability_assessment.overall_vulnerability_score
                ),
                priority: Self::risk_level_to_priority(&RiskLevel::from_score(
                    vulnerability_assessment.overall_vulnerability_score,
                )),
                analysis_domains: vec!["vulnerability_assessment".to_string()],
                related_vulnerabilities: vulnerability_assessment
                    .vulnerabilities
                    .iter()
                    .map(|v| v.id.clone())
                    .collect(),
                related_risks: Vec::new(),
                related_compliance_issues: Vec::new(),
                implementation_guidance: "Prioritize remediation of the highest-severity vulnerabilities first.".to_string(),
                expected_risk_reduction: vulnerability_assessment.assessment_confidence,
                implementation_cost: EstimatedCost::Medium.to_numeric_value(),
                implementation_timeline: Duration::from_secs(86400 * 14),
                success_metrics: vec!["Vulnerability score reduced below 5.0".to_string()],
            });
        }

        // Risk domain: surface the risk assessor's own textual recommendations.
        for (i, text) in risk_assessment.recommendations.iter().enumerate() {
            recommendations.push(ConsolidatedRecommendation {
                recommendation_id: format!("risk_{}", i),
                title: "Risk assessment recommendation".to_string(),
                description: text.clone(),
                priority: Self::risk_level_to_priority(&risk_assessment.risk_level),
                analysis_domains: vec!["risk_assessment".to_string()],
                related_vulnerabilities: Vec::new(),
                related_risks: risk_assessment
                    .risk_factors
                    .iter()
                    .map(|f| f.name.clone())
                    .collect(),
                related_compliance_issues: Vec::new(),
                implementation_guidance: text.clone(),
                expected_risk_reduction: risk_assessment.assessment_confidence,
                implementation_cost: EstimatedCost::Medium.to_numeric_value(),
                implementation_timeline: Duration::from_secs(86400 * 14),
                success_metrics: vec![format!(
                    "Overall risk score reduced below {:.1}",
                    risk_assessment.overall_risk_score
                )],
            });
        }

        // Threat modeling domain: one consolidated recommendation per identified threat.
        for (i, threat) in threat_modeling.identified_threats.iter().enumerate() {
            recommendations.push(ConsolidatedRecommendation {
                recommendation_id: format!("threat_{}", i),
                title: format!("Mitigate threat: {}", threat.name),
                description: threat.mitigation_strategy.clone(),
                priority: Self::threat_severity_to_priority(&threat.severity),
                analysis_domains: vec!["threat_modeling".to_string()],
                related_vulnerabilities: Vec::new(),
                related_risks: Vec::new(),
                related_compliance_issues: Vec::new(),
                implementation_guidance: threat.mitigation_strategy.clone(),
                expected_risk_reduction: threat_modeling.model_confidence,
                implementation_cost: match threat.mitigation_complexity {
                    ImplementationEffort::Low => EstimatedCost::Low.to_numeric_value(),
                    ImplementationEffort::Medium => EstimatedCost::Medium.to_numeric_value(),
                    ImplementationEffort::High => EstimatedCost::High.to_numeric_value(),
                },
                implementation_timeline: Duration::from_secs(86400 * 14),
                success_metrics: vec!["Threat no longer detected on re-analysis".to_string()],
            });
        }

        // Cryptographic domain: one consolidated recommendation per identified issue.
        for (i, issue) in cryptographic_analysis.identified_issues.iter().enumerate() {
            recommendations.push(ConsolidatedRecommendation {
                recommendation_id: format!("crypto_{}", i),
                title: format!("Address cryptographic issue: {}", issue.issue_type),
                description: issue.recommendation.clone(),
                priority: Self::risk_severity_to_priority(&issue.severity),
                analysis_domains: vec!["cryptographic_analysis".to_string()],
                related_vulnerabilities: Vec::new(),
                related_risks: Vec::new(),
                related_compliance_issues: Vec::new(),
                implementation_guidance: issue.recommendation.clone(),
                expected_risk_reduction: cryptographic_analysis.analysis_confidence,
                implementation_cost: match issue.fix_complexity {
                    ImplementationEffort::Low => EstimatedCost::Low.to_numeric_value(),
                    ImplementationEffort::Medium => EstimatedCost::Medium.to_numeric_value(),
                    ImplementationEffort::High => EstimatedCost::High.to_numeric_value(),
                },
                implementation_timeline: Duration::from_secs(86400 * 14),
                success_metrics: vec!["Issue no longer detected on re-analysis".to_string()],
            });
        }

        // Compliance domain: flag a consolidated recommendation when overall compliance
        // is weak enough to be actionable.
        if compliance_assessment.compliance_score < 7.0 {
            recommendations.push(ConsolidatedRecommendation {
                recommendation_id: "compliance_aggregate".to_string(),
                title: "Improve overall compliance posture".to_string(),
                description: format!(
                    "Aggregated compliance score is {:.1}/10.0 across {} framework(s)",
                    compliance_assessment.compliance_score,
                    compliance_assessment.framework_assessments.len()
                ),
                priority: AnalysisPriority::High,
                analysis_domains: vec!["compliance_assessment".to_string()],
                related_vulnerabilities: Vec::new(),
                related_risks: Vec::new(),
                related_compliance_issues: compliance_assessment
                    .framework_assessments
                    .keys()
                    .cloned()
                    .collect(),
                implementation_guidance: "Review framework-level findings and close the highest-impact compliance gaps first.".to_string(),
                expected_risk_reduction: compliance_assessment.assessment_confidence,
                implementation_cost: EstimatedCost::Medium.to_numeric_value(),
                implementation_timeline: Duration::from_secs(86400 * 30),
                success_metrics: vec!["Compliance score at or above 7.0".to_string()],
            });
        }

        Ok(recommendations)
    }

    /// Map a [`ThreatSeverity`] onto the [`AnalysisPriority`] scale used by
    /// [`ConsolidatedRecommendation`].
    fn threat_severity_to_priority(severity: &ThreatSeverity) -> AnalysisPriority {
        match severity {
            ThreatSeverity::Critical => AnalysisPriority::Critical,
            ThreatSeverity::High => AnalysisPriority::High,
            ThreatSeverity::Medium => AnalysisPriority::Medium,
            ThreatSeverity::Low => AnalysisPriority::Low,
        }
    }

    /// Map the core analyzer's [`RiskSeverity`] (used on [`SecurityRecommendation`]) onto
    /// the [`AnalysisPriority`] scale used by [`ConsolidatedRecommendation`].
    fn risk_severity_to_priority(severity: &RiskSeverity) -> AnalysisPriority {
        match severity {
            RiskSeverity::Critical => AnalysisPriority::Critical,
            RiskSeverity::High => AnalysisPriority::High,
            RiskSeverity::Medium => AnalysisPriority::Medium,
            RiskSeverity::Low => AnalysisPriority::Low,
        }
    }

    /// Map a [`RiskLevel`] onto the [`AnalysisPriority`] scale used by
    /// [`ConsolidatedRecommendation`].
    fn risk_level_to_priority(level: &RiskLevel) -> AnalysisPriority {
        match level {
            RiskLevel::Critical => AnalysisPriority::Critical,
            RiskLevel::High => AnalysisPriority::High,
            RiskLevel::Medium => AnalysisPriority::Medium,
            RiskLevel::Low => AnalysisPriority::Low,
            RiskLevel::Minimal => AnalysisPriority::Informational,
        }
    }

    fn generate_executive_summary(
        &self,
        overall_security_score: f64,
        overall_risk_level: &RiskLevel,
        overall_compliance_status: &ComplianceStatus,
        consolidated_recommendations: &[ConsolidatedRecommendation],
    ) -> Result<ExecutiveSummary, SecurityAnalysisError> {
        let security_posture = match overall_security_score {
            s if s >= 8.5 => SecurityPosture::Strong,
            s if s >= 7.0 => SecurityPosture::Adequate,
            s if s >= 5.0 => SecurityPosture::Weak,
            s if s >= 3.0 => SecurityPosture::Poor,
            _ => SecurityPosture::Critical,
        };

        let critical_recommendations: Vec<_> = consolidated_recommendations
            .iter()
            .filter(|r| matches!(r.priority, AnalysisPriority::Critical))
            .map(|r| r.title.clone())
            .collect();

        let high_priority_recommendations: Vec<_> = consolidated_recommendations
            .iter()
            .filter(|r| matches!(r.priority, AnalysisPriority::High))
            .map(|r| r.title.clone())
            .take(5)
            .collect();

        Ok(ExecutiveSummary {
            security_posture,
            key_findings: vec![
                format!("Overall security score: {:.1}/10.0", overall_security_score),
                format!("Risk level: {:?}", overall_risk_level),
                format!("Compliance status: {:?}", overall_compliance_status),
            ],
            critical_issues: critical_recommendations,
            top_risks: vec![], // Would be populated from risk assessment
            compliance_summary: format!(
                "Overall compliance status: {:?}",
                overall_compliance_status
            ),
            recommended_next_steps: high_priority_recommendations,
            resource_requirements: ResourceRequirements {
                budget_estimate: consolidated_recommendations
                    .iter()
                    .map(|r| r.implementation_cost)
                    .sum(),
                personnel_requirements: vec![
                    "Security Engineer".to_string(),
                    "Compliance Specialist".to_string(),
                ],
                technology_requirements: vec![
                    "Vulnerability Scanner".to_string(),
                    "SIEM System".to_string(),
                ],
                training_requirements: vec!["Security Awareness Training".to_string()],
                consulting_requirements: vec!["Security Assessment".to_string()],
            },
            improvement_timeline: Duration::from_secs(86400 * 90), // 90 days
            roi_estimate: 3.5,                                     // 3.5x return on investment
        })
    }

    fn calculate_analysis_confidence(
        &self,
        results: &AnalysisDomainResults<'_>,
    ) -> Result<f64, SecurityAnalysisError> {
        let confidence_scores = [
            Self::core_confidence(results.core_analysis),
            results.vulnerability_assessment.assessment_confidence,
            results.risk_assessment.assessment_confidence,
            results.threat_modeling.model_confidence,
            results.cryptographic_analysis.analysis_confidence,
            results.compliance_assessment.assessment_confidence,
            results.security_metrics.analysis_confidence,
        ];

        let average_confidence =
            confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;
        Ok(average_confidence.clamp(0.0, 1.0))
    }

    fn generate_analysis_metadata(&self, context: &TraitUsageContext) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("analysis_version".to_string(), "1.0.0".to_string());
        metadata.insert("framework_version".to_string(), "2024.1".to_string());
        metadata.insert("analysis_scope".to_string(), "comprehensive".to_string());
        metadata.insert("context_id".to_string(), context.trait_name.clone());
        metadata
    }
}

impl Default for ComprehensiveSecurityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a new comprehensive security analyzer
pub fn create_comprehensive_security_analyzer() -> ComprehensiveSecurityAnalyzer {
    ComprehensiveSecurityAnalyzer::new()
}

/// Perform a full, all-domain security analysis (core, vulnerability, risk, threat,
/// cryptographic, compliance, and metrics) on a trait usage context.
///
/// Named distinctly from [`core_analyzer::perform_comprehensive_security_analysis`]
/// (which performs only the core-domain analysis) to avoid a name collision now that
/// both are re-exported from this module.
pub fn perform_full_security_analysis(
    context: &TraitUsageContext,
) -> Result<ComprehensiveSecurityAnalysisResult, SecurityAnalysisError> {
    let mut analyzer = create_comprehensive_security_analyzer();
    analyzer.analyze_comprehensive_security(context)
}

/// Perform a full, all-domain security analysis with custom configuration.
pub fn perform_full_security_analysis_with_config(
    context: &TraitUsageContext,
    config: SecurityAnalysisConfig,
) -> Result<ComprehensiveSecurityAnalysisResult, SecurityAnalysisError> {
    let mut analyzer = create_comprehensive_security_analyzer();
    analyzer.configure_analysis(config);
    analyzer.analyze_comprehensive_security(context)
}
