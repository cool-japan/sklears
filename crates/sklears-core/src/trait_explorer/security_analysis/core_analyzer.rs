use crate::error::{Result, SklearsError};

use super::security_types::*;
// Re-exported for `security_analysis::mod`'s `pub use core_analyzer::{...}` list: these
// types are defined in `security_types` but are conceptually part of this module's public
// surface, so they need to be reachable as `core_analyzer::SecurityAnalysisError` etc.
// (a plain `use super::security_types::*;` above only brings them into *local* scope).
use super::compliance_framework::{ComplianceFrameworkManager, ComplianceStatus};
use super::crypto_analysis::{CryptographicAnalysisResult, CryptographicAnalyzer};
use super::risk_assessment::SecurityRiskAssessor;
use super::security_metrics::{SecurityMetricsCollector, SecurityMetricsResult};
pub use super::security_types::{
    RiskRecommendation, SecurityAnalysisError, SecurityAnalysisResult,
};
use super::threat_modeling::{ThreatModelingEngine, ThreatModelingResult};
use super::vulnerability_database::VulnerabilityDatabase;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// A single identified security vulnerability in a trait usage pattern.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SecurityVulnerability {
    pub id: String,
    pub category: String,
    pub severity: RiskSeverity,
    pub description: String,
    pub mitigation: String,
    pub fix_complexity: ImplementationEffort,
    pub cve_references: Vec<String>,
    pub owasp_references: Vec<String>,
    pub affected_platforms: Vec<String>,
    pub discovery_date: DateTime<Utc>,
    pub cvss_score: Option<f64>,
}

/// A single identified security risk (as opposed to a concrete vulnerability) arising
/// from a trait usage pattern.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SecurityRisk {
    pub id: String,
    pub category: String,
    pub severity: RiskSeverity,
    pub description: String,
    pub impact: String,
    pub likelihood: f64,
    pub affected_components: Vec<String>,
    pub mitigation_priority: MitigationPriority,
}

/// A recommendation to address a vulnerability, risk, threat, or cryptographic issue
/// discovered during security analysis.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SecurityRecommendation {
    pub id: String,
    pub priority: RiskSeverity,
    pub category: String,
    pub title: String,
    pub description: String,
    pub implementation_effort: ImplementationEffort,
    pub testing_requirements: Vec<String>,
    pub compliance_frameworks: Vec<String>,
    pub estimated_cost: EstimatedCost,
    pub implementation_timeline: Duration,
    pub dependencies: Vec<String>,
}

/// Metadata describing an individual security analysis run (analyzer version, timing,
/// and the analysis depth used).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SecurityAnalysisMetadata {
    pub analyzer_version: String,
    pub analysis_timestamp: DateTime<Utc>,
    pub analysis_duration: Duration,
    pub analysis_depth: AnalysisDepth,
}

/// The full result of analyzing a trait usage pattern for security concerns: known and
/// pattern-based vulnerabilities, risk factors, recommendations, compliance status, and
/// (optionally) threat modeling and cryptographic analysis results.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SecurityAnalysis {
    pub overall_risk_level: RiskLevel,
    pub vulnerabilities: Vec<SecurityVulnerability>,
    pub risk_factors: Vec<SecurityRisk>,
    pub recommendations: Vec<SecurityRecommendation>,
    pub compliance_status: ComplianceStatus,
    pub threat_analysis: Option<ThreatModelingResult>,
    pub crypto_analysis: Option<CryptographicAnalysisResult>,
    pub security_metrics: SecurityMetricsResult,
    pub analysis_timestamp: DateTime<Utc>,
    pub cache_ttl: Duration,
}

/// A cached security analysis result together with the time it was produced.
#[derive(Debug, Clone)]
struct CachedSecurityAnalysis {
    analysis: SecurityAnalysis,
    timestamp: SystemTime,
}

impl CachedSecurityAnalysis {
    /// Whether this cache entry has exceeded the default analysis cache lifetime.
    fn is_expired(&self) -> bool {
        self.is_expired_at(SystemTime::now())
    }

    /// Whether this cache entry is expired as of the given point in time.
    fn is_expired_at(&self, now: SystemTime) -> bool {
        now.duration_since(self.timestamp)
            .map(|elapsed| elapsed > DEFAULT_ANALYSIS_TIMEOUT)
            .unwrap_or(false)
    }
}

/// Main security analyzer for trait usage patterns with comprehensive vulnerability assessment,
/// risk analysis, compliance checking, and threat modeling capabilities.
///
/// # Features
///
/// - Vulnerability database integration with CVE and OWASP mappings
/// - Multi-model risk assessment with customizable risk factors
/// - Comprehensive compliance framework support
/// - Advanced threat modeling with STRIDE analysis
/// - Cryptographic security analysis including side-channel detection
/// - Real-time security monitoring and anomaly detection
/// - Automated security test generation and fuzzing recommendations
///
/// # Example
///
/// ```rust,no_run
/// use sklears_core::trait_explorer::security_analysis::{
///     create_trait_security_analyzer, TraitUsageContext,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut analyzer = create_trait_security_analyzer();
/// let context = TraitUsageContext {
///     trait_name: "MyTrait".to_string(),
///     ..Default::default()
/// };
/// let analysis = analyzer.analyze_trait_security(&context)?;
/// println!("Risk level: {:?}", analysis.overall_risk_level);
/// println!("Vulnerabilities: {}", analysis.vulnerabilities.len());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct TraitSecurityAnalyzer {
    vulnerability_database: VulnerabilityDatabase,

    risk_assessor: SecurityRiskAssessor,

    threat_modeling_engine: ThreatModelingEngine,

    crypto_analyzer: CryptographicAnalyzer,

    compliance_manager: ComplianceFrameworkManager,

    metrics_collector: SecurityMetricsCollector,

    config: SecurityAnalysisConfig,

    analysis_cache: HashMap<String, CachedSecurityAnalysis>,
}

impl TraitSecurityAnalyzer {
    /// Create a new security analyzer with default configuration.
    pub fn new() -> Self {
        Self::with_config(SecurityAnalysisConfig::default())
    }

    /// Create a new security analyzer with the specified configuration.
    pub fn with_config(config: SecurityAnalysisConfig) -> Self {
        Self {
            vulnerability_database: VulnerabilityDatabase::new(),
            risk_assessor: SecurityRiskAssessor::new(),
            threat_modeling_engine: ThreatModelingEngine::new(),
            crypto_analyzer: CryptographicAnalyzer::new(),
            compliance_manager: ComplianceFrameworkManager::new(),
            metrics_collector: SecurityMetricsCollector::new(),
            config,
            analysis_cache: HashMap::new(),
        }
    }

    /// Perform comprehensive security analysis of trait usage patterns.
    ///
    /// This is the main entry point for security analysis, providing:
    /// - Vulnerability assessment with CVE integration
    /// - Risk factor analysis using multiple risk models
    /// - Compliance checking against multiple frameworks
    /// - Threat modeling with STRIDE analysis
    /// - Cryptographic security assessment
    /// - Security recommendations and mitigation strategies
    ///
    /// # Arguments
    ///
    /// * `trait_usage` - Context describing how traits are being used
    ///
    /// # Returns
    ///
    /// Comprehensive security analysis results including risk levels, vulnerabilities,
    /// compliance status, and actionable recommendations.
    pub fn analyze_trait_security(
        &mut self,
        trait_usage: &TraitUsageContext,
    ) -> Result<SecurityAnalysis> {
        // Check cache first
        let cache_key = self.generate_cache_key(trait_usage);
        if let Some(cached) = self.analysis_cache.get(&cache_key) {
            if !cached.is_expired() {
                return Ok(cached.analysis.clone());
            }
        }

        let mut vulnerabilities = Vec::new();
        let mut risk_factors = Vec::new();
        let mut security_recommendations = Vec::new();
        let mut threat_analysis = None;
        let mut crypto_analysis = None;

        // 1. Vulnerability Assessment
        vulnerabilities.extend(self.assess_known_vulnerabilities(trait_usage)?);
        vulnerabilities.extend(self.assess_pattern_vulnerabilities(trait_usage)?);

        // 2. Risk Factor Analysis
        risk_factors.extend(self.assess_data_exposure_risks(trait_usage)?);
        risk_factors.extend(self.assess_privilege_escalation_risks(trait_usage)?);
        risk_factors.extend(self.assess_side_channel_risks(trait_usage)?);
        risk_factors.extend(self.assess_injection_risks(trait_usage)?);
        risk_factors.extend(self.assess_denial_of_service_risks(trait_usage)?);

        // 3. Threat Modeling (if enabled)
        if self.config.enable_threat_modeling {
            threat_analysis = Some(
                self.threat_modeling_engine
                    .analyze_threats(trait_usage)
                    .map_err(|e| {
                        SklearsError::AnalysisError(format!("threat modeling failed: {e}"))
                    })?,
            );
        }

        // 4. Cryptographic Analysis (if enabled)
        if self.config.enable_crypto_analysis {
            crypto_analysis = Some(
                self.crypto_analyzer
                    .analyze_cryptographic_security(trait_usage)
                    .map_err(|e| {
                        SklearsError::AnalysisError(format!("cryptographic analysis failed: {e}"))
                    })?,
            );
        }

        // 5. Generate Security Recommendations
        security_recommendations.extend(self.generate_mitigation_strategies(&vulnerabilities)?);
        security_recommendations.extend(self.generate_hardening_recommendations(trait_usage)?);
        security_recommendations.extend(self.generate_compliance_recommendations(trait_usage)?);

        if let Some(ref threat_result) = threat_analysis {
            security_recommendations
                .extend(self.generate_threat_mitigation_recommendations(threat_result)?);
        }

        if let Some(ref crypto_result) = crypto_analysis {
            security_recommendations.extend(self.generate_crypto_recommendations(crypto_result)?);
        }

        // 6. Compliance Assessment
        let compliance_status = self
            .compliance_manager
            .check_compliance_status(trait_usage)
            .map_err(|e| SklearsError::AnalysisError(format!("compliance check failed: {e}")))?;

        // 7. Calculate Overall Risk
        let overall_risk_level = self.calculate_comprehensive_risk(
            &vulnerabilities,
            &risk_factors,
            &threat_analysis,
            &crypto_analysis,
        );

        // 8. Generate Security Metrics
        let security_metrics = self
            .metrics_collector
            .collect_security_metrics(trait_usage)
            .map_err(|e| {
                SklearsError::AnalysisError(format!("security metrics collection failed: {e}"))
            })?;

        let analysis = SecurityAnalysis {
            overall_risk_level,
            vulnerabilities,
            risk_factors,
            recommendations: security_recommendations,
            compliance_status,
            threat_analysis,
            crypto_analysis,
            security_metrics,
            analysis_timestamp: Utc::now(),
            cache_ttl: self.config.cache_ttl,
        };

        // Cache the result
        self.analysis_cache.insert(
            cache_key,
            CachedSecurityAnalysis {
                analysis: analysis.clone(),
                timestamp: SystemTime::now(),
            },
        );

        Ok(analysis)
    }

    /// Assess known vulnerabilities from the vulnerability database.
    fn assess_known_vulnerabilities(
        &self,
        usage: &TraitUsageContext,
    ) -> Result<Vec<SecurityVulnerability>> {
        let mut vulnerabilities = Vec::new();

        for trait_name in &usage.traits {
            if let Some(vulns) = self.vulnerability_database.get_vulnerabilities(trait_name) {
                vulnerabilities.extend(vulns);
            }

            // Check for trait combination vulnerabilities
            if let Some(combo_vulns) = self
                .vulnerability_database
                .get_combination_vulnerabilities(&usage.traits)
            {
                vulnerabilities.extend(combo_vulns);
            }
        }

        Ok(vulnerabilities)
    }

    /// Assess vulnerabilities based on usage patterns.
    fn assess_pattern_vulnerabilities(
        &self,
        usage: &TraitUsageContext,
    ) -> Result<Vec<SecurityVulnerability>> {
        let mut vulnerabilities = Vec::new();

        // Check for unsafe trait usage patterns
        if usage.has_unsafe_operations && !usage.has_bounds_checking {
            vulnerabilities.push(SecurityVulnerability {
                id: "PATTERN-001".to_string(),
                category: "Memory Safety".to_string(),
                severity: RiskSeverity::High,
                description: "Unsafe operations without proper bounds checking".to_string(),
                mitigation: "Implement comprehensive bounds checking and validation".to_string(),
                fix_complexity: ImplementationEffort::Medium,
                cve_references: Vec::new(),
                owasp_references: vec!["A06:2021 – Vulnerable and Outdated Components".to_string()],
                affected_platforms: vec!["All".to_string()],
                discovery_date: Utc::now(),
                cvss_score: Some(7.5),
            });
        }

        // Check for serialization vulnerabilities
        if usage.has_serialization && !usage.has_input_validation {
            vulnerabilities.push(SecurityVulnerability {
                id: "PATTERN-002".to_string(),
                category: "Injection".to_string(),
                severity: RiskSeverity::High,
                description: "Serialization without input validation enables injection attacks"
                    .to_string(),
                mitigation:
                    "Implement strict input validation and sanitization for serialized data"
                        .to_string(),
                fix_complexity: ImplementationEffort::Medium,
                cve_references: Vec::new(),
                owasp_references: vec!["A03:2021 – Injection".to_string()],
                affected_platforms: vec!["All".to_string()],
                discovery_date: Utc::now(),
                cvss_score: Some(8.1),
            });
        }

        // Check for dynamic dispatch vulnerabilities
        if usage.has_dynamic_dispatch && !usage.has_type_safety_checks {
            vulnerabilities.push(SecurityVulnerability {
                id: "PATTERN-003".to_string(),
                category: "Type Safety".to_string(),
                severity: RiskSeverity::Medium,
                description: "Dynamic dispatch without type safety validation".to_string(),
                mitigation: "Implement runtime type checking and validation".to_string(),
                fix_complexity: ImplementationEffort::Medium,
                cve_references: Vec::new(),
                owasp_references: vec!["A04:2021 – Insecure Design".to_string()],
                affected_platforms: vec!["All".to_string()],
                discovery_date: Utc::now(),
                cvss_score: Some(6.1),
            });
        }

        // Check for resource management vulnerabilities
        if usage.has_resource_intensive_operations && !usage.has_resource_limits {
            vulnerabilities.push(SecurityVulnerability {
                id: "PATTERN-004".to_string(),
                category: "Resource Management".to_string(),
                severity: RiskSeverity::Medium,
                description: "Resource-intensive operations without proper limits".to_string(),
                mitigation: "Implement resource limits and monitoring".to_string(),
                fix_complexity: ImplementationEffort::Low,
                cve_references: Vec::new(),
                owasp_references: vec!["A04:2021 – Insecure Design".to_string()],
                affected_platforms: vec!["All".to_string()],
                discovery_date: Utc::now(),
                cvss_score: Some(5.3),
            });
        }

        Ok(vulnerabilities)
    }

    /// Assess data exposure risks based on trait usage context.
    fn assess_data_exposure_risks(&self, usage: &TraitUsageContext) -> Result<Vec<SecurityRisk>> {
        let mut risks = Vec::new();

        if usage.handles_sensitive_data {
            let risk_level = if usage.has_encryption {
                RiskSeverity::Low
            } else if usage.has_access_controls {
                RiskSeverity::Medium
            } else {
                RiskSeverity::High
            };

            risks.push(SecurityRisk {
                id: "RISK-DATA-001".to_string(),
                category: "Data Exposure".to_string(),
                severity: risk_level,
                description: "Trait handles sensitive data with potential exposure risks"
                    .to_string(),
                impact: "Potential data leakage, privacy violations, or unauthorized access"
                    .to_string(),
                likelihood: if usage.has_encryption { 0.2 } else { 0.7 },
                affected_components: usage.traits.clone(),
                mitigation_priority: if usage.handles_personal_data {
                    MitigationPriority::Critical
                } else {
                    MitigationPriority::High
                },
            });
        }

        if usage.handles_personal_data && !usage.has_data_anonymization {
            risks.push(SecurityRisk {
                id: "RISK-DATA-002".to_string(),
                category: "Privacy".to_string(),
                severity: RiskSeverity::High,
                description: "Personal data handling without anonymization".to_string(),
                impact: "GDPR violations, privacy breaches, regulatory penalties".to_string(),
                likelihood: 0.8,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Critical,
            });
        }

        if usage.has_serialization && usage.handles_sensitive_data {
            risks.push(SecurityRisk {
                id: "RISK-DATA-003".to_string(),
                category: "Data Exposure".to_string(),
                severity: RiskSeverity::Medium,
                description: "Serialization of sensitive data without protection".to_string(),
                impact: "Data exposure through serialization channels".to_string(),
                likelihood: 0.5,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::High,
            });
        }

        Ok(risks)
    }

    /// Assess privilege escalation risks.
    fn assess_privilege_escalation_risks(
        &self,
        usage: &TraitUsageContext,
    ) -> Result<Vec<SecurityRisk>> {
        let mut risks = Vec::new();

        if usage.requires_elevated_privileges {
            risks.push(SecurityRisk {
                id: "RISK-PRIV-001".to_string(),
                category: "Privilege Escalation".to_string(),
                severity: RiskSeverity::Medium,
                description: "Trait requires elevated privileges for operation".to_string(),
                impact: "Potential for privilege escalation attacks and unauthorized system access"
                    .to_string(),
                likelihood: if usage.has_access_controls { 0.3 } else { 0.6 },
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::High,
            });
        }

        if usage.has_dynamic_dispatch && !usage.has_type_safety_checks {
            risks.push(SecurityRisk {
                id: "RISK-PRIV-002".to_string(),
                category: "Type Confusion".to_string(),
                severity: RiskSeverity::Medium,
                description: "Dynamic dispatch without type safety checks".to_string(),
                impact: "Type confusion attacks leading to privilege escalation".to_string(),
                likelihood: 0.4,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Medium,
            });
        }

        if usage.has_unsafe_operations && !usage.has_privilege_separation {
            risks.push(SecurityRisk {
                id: "RISK-PRIV-003".to_string(),
                category: "Unsafe Operations".to_string(),
                severity: RiskSeverity::High,
                description: "Unsafe operations without privilege separation".to_string(),
                impact: "Memory corruption leading to privilege escalation".to_string(),
                likelihood: 0.5,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Critical,
            });
        }

        Ok(risks)
    }

    /// Assess side channel attack risks.
    fn assess_side_channel_risks(&self, usage: &TraitUsageContext) -> Result<Vec<SecurityRisk>> {
        let mut risks = Vec::new();

        if usage.has_timing_dependencies {
            risks.push(SecurityRisk {
                id: "RISK-SIDE-001".to_string(),
                category: "Side Channel".to_string(),
                severity: RiskSeverity::Medium,
                description: "Trait implementation may be vulnerable to timing attacks".to_string(),
                impact: "Information disclosure through timing analysis".to_string(),
                likelihood: 0.3,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Medium,
            });
        }

        if usage.has_cryptographic_operations && !usage.has_constant_time_operations {
            risks.push(SecurityRisk {
                id: "RISK-SIDE-002".to_string(),
                category: "Cryptographic Side Channel".to_string(),
                severity: RiskSeverity::High,
                description: "Cryptographic operations without constant-time guarantees"
                    .to_string(),
                impact: "Key recovery through side-channel analysis".to_string(),
                likelihood: 0.5,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Critical,
            });
        }

        if usage.has_memory_allocation_patterns {
            risks.push(SecurityRisk {
                id: "RISK-SIDE-003".to_string(),
                category: "Memory Side Channel".to_string(),
                severity: RiskSeverity::Low,
                description: "Memory allocation patterns may leak information".to_string(),
                impact: "Information disclosure through memory allocation analysis".to_string(),
                likelihood: 0.2,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Low,
            });
        }

        Ok(risks)
    }

    /// Assess injection attack risks.
    fn assess_injection_risks(&self, usage: &TraitUsageContext) -> Result<Vec<SecurityRisk>> {
        let mut risks = Vec::new();

        if usage.has_user_input && !usage.has_input_validation {
            risks.push(SecurityRisk {
                id: "RISK-INJ-001".to_string(),
                category: "Injection".to_string(),
                severity: RiskSeverity::High,
                description: "User input handling without proper validation".to_string(),
                impact: "Code injection, data corruption, or system compromise".to_string(),
                likelihood: 0.7,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Critical,
            });
        }

        if usage.has_sql_operations && !usage.has_parameterized_queries {
            risks.push(SecurityRisk {
                id: "RISK-INJ-002".to_string(),
                category: "SQL Injection".to_string(),
                severity: RiskSeverity::Critical,
                description: "SQL operations without parameterized queries".to_string(),
                impact: "Database compromise, data theft, or unauthorized modifications"
                    .to_string(),
                likelihood: 0.8,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Critical,
            });
        }

        if usage.has_serialization && !usage.has_input_validation {
            risks.push(SecurityRisk {
                id: "RISK-INJ-003".to_string(),
                category: "Deserialization".to_string(),
                severity: RiskSeverity::High,
                description: "Deserialization without input validation".to_string(),
                impact: "Remote code execution through malicious serialized data".to_string(),
                likelihood: 0.6,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Critical,
            });
        }

        Ok(risks)
    }

    /// Assess denial of service risks.
    fn assess_denial_of_service_risks(
        &self,
        usage: &TraitUsageContext,
    ) -> Result<Vec<SecurityRisk>> {
        let mut risks = Vec::new();

        if usage.has_resource_intensive_operations && !usage.has_rate_limiting {
            risks.push(SecurityRisk {
                id: "RISK-DOS-001".to_string(),
                category: "Denial of Service".to_string(),
                severity: RiskSeverity::Medium,
                description: "Resource-intensive operations without rate limiting".to_string(),
                impact: "Service unavailability through resource exhaustion".to_string(),
                likelihood: 0.5,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Medium,
            });
        }

        if usage.has_unbounded_recursion {
            risks.push(SecurityRisk {
                id: "RISK-DOS-002".to_string(),
                category: "Stack Overflow".to_string(),
                severity: RiskSeverity::High,
                description: "Potential unbounded recursion leading to stack overflow".to_string(),
                impact: "Application crash and service denial".to_string(),
                likelihood: 0.6,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::High,
            });
        }

        if usage.has_memory_allocation_patterns && !usage.has_memory_limits {
            risks.push(SecurityRisk {
                id: "RISK-DOS-003".to_string(),
                category: "Memory Exhaustion".to_string(),
                severity: RiskSeverity::Medium,
                description: "Uncontrolled memory allocation patterns".to_string(),
                impact: "Memory exhaustion leading to service denial".to_string(),
                likelihood: 0.4,
                affected_components: usage.traits.clone(),
                mitigation_priority: MitigationPriority::Medium,
            });
        }

        Ok(risks)
    }

    /// Generate mitigation strategies for identified vulnerabilities.
    fn generate_mitigation_strategies(
        &self,
        vulnerabilities: &[SecurityVulnerability],
    ) -> Result<Vec<SecurityRecommendation>> {
        let mut recommendations = Vec::new();

        for vuln in vulnerabilities {
            recommendations.push(SecurityRecommendation {
                id: format!("REC-MIT-{}", vuln.id),
                priority: vuln.severity.clone(),
                category: "Vulnerability Mitigation".to_string(),
                title: format!("Address {}: {}", vuln.category, vuln.id),
                description: vuln.mitigation.clone(),
                implementation_effort: vuln.fix_complexity.clone(),
                testing_requirements: vec![
                    "Security regression testing".to_string(),
                    "Penetration testing".to_string(),
                    "Code review with security focus".to_string(),
                ],
                compliance_frameworks: self.get_relevant_compliance_frameworks(&vuln.category),
                estimated_cost: self.estimate_mitigation_cost(&vuln.fix_complexity),
                implementation_timeline: self
                    .estimate_implementation_timeline(&vuln.fix_complexity),
                dependencies: Vec::new(),
            });
        }

        Ok(recommendations)
    }

    /// Generate hardening recommendations based on usage context.
    fn generate_hardening_recommendations(
        &self,
        usage: &TraitUsageContext,
    ) -> Result<Vec<SecurityRecommendation>> {
        let mut recommendations = Vec::new();

        if usage.handles_sensitive_data && !usage.has_encryption {
            recommendations.push(SecurityRecommendation {
                id: "REC-HARD-001".to_string(),
                priority: RiskSeverity::High,
                category: "Data Protection".to_string(),
                title: "Implement encryption for sensitive data".to_string(),
                description: "Implement encryption for sensitive data at rest and in transit using industry-standard algorithms".to_string(),
                implementation_effort: ImplementationEffort::Medium,
                testing_requirements: vec![
                    "Encryption validation".to_string(),
                    "Key management testing".to_string(),
                    "Performance impact assessment".to_string(),
                ],
                compliance_frameworks: vec!["GDPR".to_string(), "HIPAA".to_string(), "SOC 2".to_string()],
                estimated_cost: EstimatedCost::Medium,
                implementation_timeline: Duration::from_secs(86400 * 14), // 2 weeks
                dependencies: vec!["Cryptographic library integration".to_string()],
            });
        }

        if !usage.has_input_validation {
            recommendations.push(SecurityRecommendation {
                id: "REC-HARD-002".to_string(),
                priority: RiskSeverity::High,
                category: "Input Validation".to_string(),
                title: "Implement comprehensive input validation".to_string(),
                description: "Implement comprehensive input validation and sanitization for all user-provided data".to_string(),
                implementation_effort: ImplementationEffort::Low,
                testing_requirements: vec![
                    "Boundary value testing".to_string(),
                    "Malformed input testing".to_string(),
                    "Fuzzing validation".to_string(),
                ],
                compliance_frameworks: vec!["OWASP".to_string(), "NIST".to_string()],
                estimated_cost: EstimatedCost::Low,
                implementation_timeline: Duration::from_secs(86400 * 7), // 1 week
                dependencies: Vec::new(),
            });
        }

        if !usage.has_audit_logging {
            recommendations.push(SecurityRecommendation {
                id: "REC-HARD-003".to_string(),
                priority: RiskSeverity::Medium,
                category: "Audit and Monitoring".to_string(),
                title: "Implement comprehensive audit logging".to_string(),
                description: "Implement detailed audit logging for all security-relevant operations and access attempts".to_string(),
                implementation_effort: ImplementationEffort::Medium,
                testing_requirements: vec![
                    "Log completeness validation".to_string(),
                    "Log integrity verification".to_string(),
                    "Performance impact assessment".to_string(),
                ],
                compliance_frameworks: vec!["SOC 2".to_string(), "ISO 27001".to_string()],
                estimated_cost: EstimatedCost::Medium,
                implementation_timeline: Duration::from_secs(86400 * 10), // 10 days
                dependencies: vec!["Logging infrastructure setup".to_string()],
            });
        }

        if usage.has_cryptographic_operations && !usage.has_secure_key_management {
            recommendations.push(SecurityRecommendation {
                id: "REC-HARD-004".to_string(),
                priority: RiskSeverity::High,
                category: "Key Management".to_string(),
                title: "Implement secure key management".to_string(),
                description: "Implement secure key generation, storage, and rotation mechanisms"
                    .to_string(),
                implementation_effort: ImplementationEffort::High,
                testing_requirements: vec![
                    "Key lifecycle validation".to_string(),
                    "Security hardware integration testing".to_string(),
                    "Key rotation validation".to_string(),
                ],
                compliance_frameworks: vec![
                    "FIPS 140-2".to_string(),
                    "Common Criteria".to_string(),
                ],
                estimated_cost: EstimatedCost::High,
                implementation_timeline: Duration::from_secs(86400 * 21), // 3 weeks
                dependencies: vec![
                    "HSM integration".to_string(),
                    "Key management infrastructure".to_string(),
                ],
            });
        }

        Ok(recommendations)
    }

    /// Generate compliance-specific recommendations.
    fn generate_compliance_recommendations(
        &self,
        usage: &TraitUsageContext,
    ) -> Result<Vec<SecurityRecommendation>> {
        let mut recommendations = Vec::new();

        if usage.handles_personal_data {
            recommendations.push(SecurityRecommendation {
                id: "REC-COMP-001".to_string(),
                priority: RiskSeverity::Critical,
                category: "GDPR Compliance".to_string(),
                title: "Implement GDPR privacy controls".to_string(),
                description: "Implement comprehensive GDPR privacy controls including consent management, data anonymization, and right to erasure".to_string(),
                implementation_effort: ImplementationEffort::High,
                testing_requirements: vec![
                    "Privacy impact assessment".to_string(),
                    "Data processing audit".to_string(),
                    "Consent mechanism validation".to_string(),
                ],
                compliance_frameworks: vec!["GDPR".to_string()],
                estimated_cost: EstimatedCost::High,
                implementation_timeline: Duration::from_secs(86400 * 30), // 30 days
                dependencies: vec!["Legal review".to_string(), "Privacy framework integration".to_string()],
            });
        }

        if usage.has_cryptographic_operations {
            recommendations.push(SecurityRecommendation {
                id: "REC-COMP-002".to_string(),
                priority: RiskSeverity::Medium,
                category: "FIPS Compliance".to_string(),
                title: "Implement FIPS 140-2 compliant cryptography".to_string(),
                description: "Ensure all cryptographic operations comply with FIPS 140-2 standards"
                    .to_string(),
                implementation_effort: ImplementationEffort::Medium,
                testing_requirements: vec![
                    "FIPS validation testing".to_string(),
                    "Cryptographic module certification".to_string(),
                    "Algorithm compliance verification".to_string(),
                ],
                compliance_frameworks: vec!["FIPS 140-2".to_string()],
                estimated_cost: EstimatedCost::Medium,
                implementation_timeline: Duration::from_secs(86400 * 14), // 2 weeks
                dependencies: vec!["FIPS-certified library integration".to_string()],
            });
        }

        Ok(recommendations)
    }

    /// Generate threat-specific mitigation recommendations.
    fn generate_threat_mitigation_recommendations(
        &self,
        threat_analysis: &ThreatModelingResult,
    ) -> Result<Vec<SecurityRecommendation>> {
        let mut recommendations = Vec::new();

        for threat in &threat_analysis.identified_threats {
            recommendations.push(SecurityRecommendation {
                id: format!("REC-THREAT-{}", threat.id),
                priority: self.threat_severity_to_risk_severity(&threat.severity),
                category: "Threat Mitigation".to_string(),
                title: format!("Mitigate threat: {}", threat.name),
                description: threat.mitigation_strategy.clone(),
                implementation_effort: threat.mitigation_complexity.clone(),
                testing_requirements: vec![
                    "Threat simulation testing".to_string(),
                    "Attack scenario validation".to_string(),
                ],
                compliance_frameworks: Vec::new(),
                estimated_cost: self.estimate_threat_mitigation_cost(&threat.mitigation_complexity),
                implementation_timeline: self
                    .estimate_threat_mitigation_timeline(&threat.mitigation_complexity),
                dependencies: threat.mitigation_dependencies.clone(),
            });
        }

        Ok(recommendations)
    }

    /// Generate cryptographic-specific recommendations.
    fn generate_crypto_recommendations(
        &self,
        crypto_analysis: &CryptographicAnalysisResult,
    ) -> Result<Vec<SecurityRecommendation>> {
        let mut recommendations = Vec::new();

        for issue in &crypto_analysis.identified_issues {
            recommendations.push(SecurityRecommendation {
                id: format!("REC-CRYPTO-{}", issue.id),
                priority: issue.severity.clone(),
                category: "Cryptographic Security".to_string(),
                title: format!("Address cryptographic issue: {}", issue.issue_type),
                description: issue.recommendation.clone(),
                implementation_effort: issue.fix_complexity.clone(),
                testing_requirements: vec![
                    "Cryptographic validation".to_string(),
                    "Side-channel analysis".to_string(),
                    "Constant-time verification".to_string(),
                ],
                compliance_frameworks: vec![
                    "FIPS 140-2".to_string(),
                    "Common Criteria".to_string(),
                ],
                estimated_cost: self.estimate_crypto_fix_cost(&issue.fix_complexity),
                implementation_timeline: self.estimate_crypto_fix_timeline(&issue.fix_complexity),
                dependencies: issue.dependencies.clone(),
            });
        }

        Ok(recommendations)
    }

    /// Calculate comprehensive risk level considering all factors.
    fn calculate_comprehensive_risk(
        &self,
        vulnerabilities: &[SecurityVulnerability],
        risk_factors: &[SecurityRisk],
        threat_analysis: &Option<ThreatModelingResult>,
        crypto_analysis: &Option<CryptographicAnalysisResult>,
    ) -> RiskLevel {
        let mut risk_score: f64 = 0.0;

        // Weight vulnerabilities by severity and CVSS score
        for vuln in vulnerabilities {
            let base_score = match vuln.severity {
                RiskSeverity::Critical => 4.0,
                RiskSeverity::High => 3.0,
                RiskSeverity::Medium => 2.0,
                RiskSeverity::Low => 1.0,
            };

            let cvss_multiplier = vuln.cvss_score.map(|score| score / 10.0).unwrap_or(1.0);
            risk_score += base_score * cvss_multiplier;
        }

        // Weight risk factors by severity and likelihood
        for risk in risk_factors {
            let base_score = match risk.severity {
                RiskSeverity::Critical => 4.0,
                RiskSeverity::High => 3.0,
                RiskSeverity::Medium => 2.0,
                RiskSeverity::Low => 1.0,
            };

            risk_score += base_score * risk.likelihood;
        }

        // Include threat analysis score
        if let Some(ref threat_result) = threat_analysis {
            risk_score += threat_result.overall_risk_score * 0.5; // Weight threat analysis at 50%
        }

        // Include cryptographic analysis score
        if let Some(ref crypto_result) = crypto_analysis {
            risk_score += crypto_result.risk_score * 0.3; // Weight crypto analysis at 30%
        }

        // Convert to risk level
        match risk_score {
            x if x >= 15.0 => RiskLevel::Critical,
            x if x >= 10.0 => RiskLevel::High,
            x if x >= 6.0 => RiskLevel::Medium,
            x if x >= 2.0 => RiskLevel::Low,
            _ => RiskLevel::Minimal,
        }
    }

    /// Generate cache key for security analysis.
    fn generate_cache_key(&self, usage: &TraitUsageContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        usage.hash(&mut hasher);
        format!("security_analysis_{:x}", hasher.finish())
    }

    /// Clear expired entries from analysis cache.
    pub fn cleanup_cache(&mut self) {
        let now = SystemTime::now();
        self.analysis_cache
            .retain(|_, cached| !cached.is_expired_at(now));
    }

    /// Get cache statistics for monitoring.
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let total_entries = self.analysis_cache.len();
        let expired_entries = self
            .analysis_cache
            .values()
            .filter(|cached| cached.is_expired())
            .count();

        CacheStatistics {
            total_entries,
            expired_entries,
            hit_rate: 0.0, // Would need tracking for actual hit rate
        }
    }

    /// Helper methods for cost and timeline estimation
    fn estimate_mitigation_cost(&self, complexity: &ImplementationEffort) -> EstimatedCost {
        match complexity {
            ImplementationEffort::Low => EstimatedCost::Low,
            ImplementationEffort::Medium => EstimatedCost::Medium,
            ImplementationEffort::High => EstimatedCost::High,
        }
    }

    fn estimate_implementation_timeline(&self, complexity: &ImplementationEffort) -> Duration {
        match complexity {
            ImplementationEffort::Low => Duration::from_secs(86400 * 3), // 3 days
            ImplementationEffort::Medium => Duration::from_secs(86400 * 10), // 10 days
            ImplementationEffort::High => Duration::from_secs(86400 * 30), // 30 days
        }
    }

    fn threat_severity_to_risk_severity(&self, severity: &ThreatSeverity) -> RiskSeverity {
        match severity {
            ThreatSeverity::Critical => RiskSeverity::Critical,
            ThreatSeverity::High => RiskSeverity::High,
            ThreatSeverity::Medium => RiskSeverity::Medium,
            ThreatSeverity::Low => RiskSeverity::Low,
        }
    }

    fn estimate_threat_mitigation_cost(&self, complexity: &ImplementationEffort) -> EstimatedCost {
        self.estimate_mitigation_cost(complexity)
    }

    fn estimate_threat_mitigation_timeline(&self, complexity: &ImplementationEffort) -> Duration {
        self.estimate_implementation_timeline(complexity)
    }

    fn estimate_crypto_fix_cost(&self, complexity: &ImplementationEffort) -> EstimatedCost {
        self.estimate_mitigation_cost(complexity)
    }

    fn estimate_crypto_fix_timeline(&self, complexity: &ImplementationEffort) -> Duration {
        self.estimate_implementation_timeline(complexity)
    }

    fn get_relevant_compliance_frameworks(&self, category: &str) -> Vec<String> {
        match category.to_lowercase().as_str() {
            "data exposure" | "privacy" => vec!["GDPR".to_string(), "HIPAA".to_string()],
            "injection" | "memory safety" => vec!["OWASP".to_string(), "NIST".to_string()],
            "cryptographic" => vec!["FIPS 140-2".to_string(), "Common Criteria".to_string()],
            _ => vec!["ISO 27001".to_string()],
        }
    }

    /// Update configuration for the security analyzer.
    pub fn update_config(&mut self, config: SecurityAnalysisConfig) {
        self.config = config;
        self.analysis_cache.clear(); // Clear cache when config changes
    }

    /// Get current configuration.
    pub fn get_config(&self) -> &SecurityAnalysisConfig {
        &self.config
    }

    /// Get vulnerability database for external access.
    pub fn get_vulnerability_database(&self) -> &VulnerabilityDatabase {
        &self.vulnerability_database
    }

    /// Get risk assessor for external access.
    pub fn get_risk_assessor(&self) -> &SecurityRiskAssessor {
        &self.risk_assessor
    }

    /// Get threat modeling engine for external access.
    pub fn get_threat_modeling_engine(&self) -> &ThreatModelingEngine {
        &self.threat_modeling_engine
    }

    /// Get cryptographic analyzer for external access.
    pub fn get_crypto_analyzer(&self) -> &CryptographicAnalyzer {
        &self.crypto_analyzer
    }

    /// Get compliance manager for external access.
    pub fn get_compliance_manager(&self) -> &ComplianceFrameworkManager {
        &self.compliance_manager
    }

    /// Get metrics collector for external access.
    pub fn get_metrics_collector(&self) -> &SecurityMetricsCollector {
        &self.metrics_collector
    }
}

impl Default for TraitSecurityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub expired_entries: usize,
    pub hit_rate: f64,
}

/// Create a new [`TraitSecurityAnalyzer`] with default configuration.
///
/// Thin constructor wrapper kept for API-surface consistency with the other
/// `create_*` factory functions in `trait_explorer` (e.g.
/// [`super::vulnerability_database::create_vulnerability_database`],
/// [`super::risk_assessment::create_security_risk_assessor`]).
pub fn create_trait_security_analyzer() -> TraitSecurityAnalyzer {
    TraitSecurityAnalyzer::new()
}

/// Perform a comprehensive security analysis of a trait usage pattern using a
/// freshly-constructed [`TraitSecurityAnalyzer`] with default configuration.
pub fn perform_comprehensive_security_analysis(
    trait_usage: &TraitUsageContext,
) -> Result<SecurityAnalysis> {
    let mut analyzer = create_trait_security_analyzer();
    analyzer.analyze_trait_security(trait_usage)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: build a minimal, high-risk trait usage context, run the full
    /// comprehensive security analysis entry point, and assert that every
    /// top-level section of the result is populated without panicking.
    #[test]
    fn test_comprehensive_security_analysis_smoke() {
        let context = TraitUsageContext {
            trait_name: "Serialize".to_string(),
            traits: vec!["Serialize".to_string(), "Clone".to_string()],
            handles_sensitive_data: true,
            has_serialization: true,
            has_input_validation: false,
            has_unsafe_operations: true,
            requires_elevated_privileges: true,
            has_cryptographic_operations: true,
            ..Default::default()
        };

        let result = perform_comprehensive_security_analysis(&context);
        assert!(result.is_ok(), "analysis should succeed: {result:?}");

        let analysis = result.expect("analysis should succeed");
        assert!(
            !analysis.vulnerabilities.is_empty(),
            "expected at least one vulnerability for an unsafe, unvalidated Serialize/Clone context"
        );
        assert!(
            !analysis.risk_factors.is_empty(),
            "expected at least one risk factor for a sensitive-data, elevated-privilege context"
        );
        assert!(
            !analysis.recommendations.is_empty(),
            "expected at least one security recommendation"
        );
        assert!(analysis.security_metrics.overall_security_score >= 0.0);
    }

    #[test]
    fn test_create_trait_security_analyzer() {
        let analyzer = create_trait_security_analyzer();
        assert_eq!(analyzer.get_cache_stats().total_entries, 0);
    }
}
