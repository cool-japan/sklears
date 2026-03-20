//! Core compliance context traits and foundational types
//!
//! This module provides the foundational types and traits for the compliance
//! management system, including the main ComplianceContext struct and its
//! core configuration and state management.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock, Mutex},
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextError, ContextResult,
    ContextMetadata, ContextEvent,
};

// Import from other compliance modules
use crate::compliance_governance::DataGovernanceManager;
use crate::compliance_regulatory::RegulatoryFrameworkManager;
use crate::compliance_audit::ComplianceAuditor;
use crate::compliance_policy::CompliancePolicyEngine;
use crate::compliance_consent::ConsentManager;
use crate::compliance_retention::DataRetentionManager;
use crate::compliance_breach::BreachDetector;
use crate::compliance_metrics::ComplianceMetrics;

/// Compliance context for regulatory and data governance management
#[derive(Debug)]
pub struct ComplianceContext {
    /// Context identifier
    pub id: String,
    /// Compliance state
    pub state: Arc<RwLock<ComplianceState>>,
    /// Data governance manager
    pub data_governance: Arc<RwLock<DataGovernanceManager>>,
    /// Regulatory framework manager
    pub regulatory_manager: Arc<RwLock<RegulatoryFrameworkManager>>,
    /// Compliance auditor
    pub auditor: Arc<Mutex<ComplianceAuditor>>,
    /// Policy engine
    pub policy_engine: Arc<RwLock<CompliancePolicyEngine>>,
    /// Consent manager
    pub consent_manager: Arc<RwLock<ConsentManager>>,
    /// Data retention manager
    pub retention_manager: Arc<Mutex<DataRetentionManager>>,
    /// Breach detector
    pub breach_detector: Arc<Mutex<BreachDetector>>,
    /// Compliance metrics
    pub metrics: Arc<Mutex<ComplianceMetrics>>,
    /// Configuration
    pub config: Arc<RwLock<ComplianceConfig>>,
    /// Created timestamp
    pub created_at: SystemTime,
}

/// Compliance context states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceState {
    /// Compliance context is initializing
    Initializing,
    /// Compliance context is active
    Active,
    /// Compliance context is in audit mode
    Audit,
    /// Compliance context is non-compliant
    NonCompliant,
    /// Compliance context is under investigation
    Investigation,
    /// Compliance context is disabled
    Disabled,
    /// Compliance context is in remediation mode
    Remediation,
}

impl Display for ComplianceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplianceState::Initializing => write!(f, "initializing"),
            ComplianceState::Active => write!(f, "active"),
            ComplianceState::Audit => write!(f, "audit"),
            ComplianceState::NonCompliant => write!(f, "non_compliant"),
            ComplianceState::Investigation => write!(f, "investigation"),
            ComplianceState::Disabled => write!(f, "disabled"),
            ComplianceState::Remediation => write!(f, "remediation"),
        }
    }
}

/// Compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Enabled regulatory frameworks
    pub enabled_frameworks: HashSet<RegulatoryFramework>,
    /// Data governance settings
    pub data_governance_settings: DataGovernanceSettings,
    /// Audit settings
    pub audit_settings: AuditSettings,
    /// Retention settings
    pub retention_settings: RetentionSettings,
    /// Breach detection settings
    pub breach_detection_settings: BreachDetectionSettings,
    /// Consent management settings
    pub consent_settings: ConsentSettings,
    /// Reporting settings
    pub reporting_settings: ReportingSettings,
    /// Custom configuration
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        let mut enabled_frameworks = HashSet::new();
        enabled_frameworks.insert(RegulatoryFramework::Gdpr);

        Self {
            enabled_frameworks,
            data_governance_settings: DataGovernanceSettings::default(),
            audit_settings: AuditSettings::default(),
            retention_settings: RetentionSettings::default(),
            breach_detection_settings: BreachDetectionSettings::default(),
            consent_settings: ConsentSettings::default(),
            reporting_settings: ReportingSettings::default(),
            custom: HashMap::new(),
        }
    }
}

/// Regulatory frameworks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegulatoryFramework {
    /// General Data Protection Regulation (EU)
    Gdpr,
    /// California Consumer Privacy Act
    Ccpa,
    /// Health Insurance Portability and Accountability Act (US)
    Hipaa,
    /// Sarbanes-Oxley Act (US)
    Sox,
    /// Payment Card Industry Data Security Standard
    PciDss,
    /// ISO 27001 Information Security Management
    Iso27001,
    /// ISO 27002 Code of Practice for Information Security
    Iso27002,
    /// NIST Cybersecurity Framework
    NistCsf,
    /// Brazilian General Data Protection Law
    Lgpd,
    /// Personal Information Protection Act (Canada)
    Pipeda,
    /// Data Protection Act (UK)
    Dpa,
    /// Custom regulatory framework
    Custom(String),
}

impl Display for RegulatoryFramework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegulatoryFramework::Gdpr => write!(f, "GDPR"),
            RegulatoryFramework::Ccpa => write!(f, "CCPA"),
            RegulatoryFramework::Hipaa => write!(f, "HIPAA"),
            RegulatoryFramework::Sox => write!(f, "SOX"),
            RegulatoryFramework::PciDss => write!(f, "PCI-DSS"),
            RegulatoryFramework::Iso27001 => write!(f, "ISO-27001"),
            RegulatoryFramework::Iso27002 => write!(f, "ISO-27002"),
            RegulatoryFramework::NistCsf => write!(f, "NIST-CSF"),
            RegulatoryFramework::Lgpd => write!(f, "LGPD"),
            RegulatoryFramework::Pipeda => write!(f, "PIPEDA"),
            RegulatoryFramework::Dpa => write!(f, "DPA"),
            RegulatoryFramework::Custom(name) => write!(f, "CUSTOM-{}", name),
        }
    }
}

/// Data governance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataGovernanceSettings {
    /// Enable data classification
    pub enable_classification: bool,
    /// Enable data lineage tracking
    pub enable_lineage: bool,
    /// Enable data quality monitoring
    pub enable_quality_monitoring: bool,
    /// Classification refresh interval
    pub classification_interval: Duration,
    /// Data catalog settings
    pub catalog_settings: DataCatalogSettings,
}

impl Default for DataGovernanceSettings {
    fn default() -> Self {
        Self {
            enable_classification: true,
            enable_lineage: true,
            enable_quality_monitoring: true,
            classification_interval: Duration::from_secs(24 * 60 * 60), // Daily
            catalog_settings: DataCatalogSettings::default(),
        }
    }
}

/// Data catalog settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCatalogSettings {
    /// Enable metadata collection
    pub enable_metadata: bool,
    /// Automatic discovery
    pub auto_discovery: bool,
    /// Schema validation
    pub schema_validation: bool,
    /// Profiling enabled
    pub profiling_enabled: bool,
}

impl Default for DataCatalogSettings {
    fn default() -> Self {
        Self {
            enable_metadata: true,
            auto_discovery: true,
            schema_validation: true,
            profiling_enabled: true,
        }
    }
}

/// Audit settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSettings {
    /// Enable comprehensive auditing
    pub enable_auditing: bool,
    /// Audit log retention period
    pub audit_retention: Duration,
    /// Real-time compliance checking
    pub real_time_checking: bool,
    /// Automated reporting
    pub automated_reporting: bool,
    /// Report generation interval
    pub report_interval: Duration,
}

impl Default for AuditSettings {
    fn default() -> Self {
        Self {
            enable_auditing: true,
            audit_retention: Duration::from_secs(7 * 365 * 24 * 60 * 60), // 7 years
            real_time_checking: true,
            automated_reporting: true,
            report_interval: Duration::from_secs(30 * 24 * 60 * 60), // Monthly
        }
    }
}

/// Retention settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionSettings {
    /// Enable automatic data retention
    pub enable_retention: bool,
    /// Default retention period
    pub default_retention_period: Duration,
    /// Enable legal hold
    pub enable_legal_hold: bool,
    /// Deletion verification
    pub verify_deletion: bool,
    /// Retention policy enforcement
    pub enforce_policies: bool,
}

impl Default for RetentionSettings {
    fn default() -> Self {
        Self {
            enable_retention: true,
            default_retention_period: Duration::from_secs(7 * 365 * 24 * 60 * 60), // 7 years
            enable_legal_hold: true,
            verify_deletion: true,
            enforce_policies: true,
        }
    }
}

/// Breach detection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachDetectionSettings {
    /// Enable breach detection
    pub enable_detection: bool,
    /// Detection sensitivity
    pub sensitivity_level: SensitivityLevel,
    /// Automatic notification
    pub auto_notification: bool,
    /// Investigation timeout
    pub investigation_timeout: Duration,
    /// Escalation rules
    pub escalation_enabled: bool,
}

impl Default for BreachDetectionSettings {
    fn default() -> Self {
        Self {
            enable_detection: true,
            sensitivity_level: SensitivityLevel::Medium,
            auto_notification: true,
            investigation_timeout: Duration::from_secs(72 * 60 * 60), // 72 hours
            escalation_enabled: true,
        }
    }
}

/// Sensitivity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensitivityLevel {
    /// Low sensitivity
    Low,
    /// Medium sensitivity
    Medium,
    /// High sensitivity
    High,
    /// Maximum sensitivity
    Maximum,
}

/// Consent management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentSettings {
    /// Enable consent management
    pub enable_consent: bool,
    /// Consent expiration period
    pub consent_expiry: Duration,
    /// Require explicit consent
    pub require_explicit_consent: bool,
    /// Enable consent withdrawal
    pub enable_withdrawal: bool,
    /// Consent audit trail
    pub audit_trail: bool,
}

impl Default for ConsentSettings {
    fn default() -> Self {
        Self {
            enable_consent: true,
            consent_expiry: Duration::from_secs(2 * 365 * 24 * 60 * 60), // 2 years
            require_explicit_consent: true,
            enable_withdrawal: true,
            audit_trail: true,
        }
    }
}

/// Reporting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingSettings {
    /// Enable automated reporting
    pub automated_reports: bool,
    /// Report formats
    pub report_formats: Vec<ReportFormat>,
    /// Report distribution
    pub distribution_channels: Vec<String>,
    /// Dashboard enabled
    pub dashboard_enabled: bool,
    /// Real-time metrics
    pub real_time_metrics: bool,
}

impl Default for ReportingSettings {
    fn default() -> Self {
        Self {
            automated_reports: true,
            report_formats: vec![ReportFormat::Pdf, ReportFormat::Json],
            distribution_channels: vec!["email".to_string()],
            dashboard_enabled: true,
            real_time_metrics: true,
        }
    }
}

/// Report formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// PDF format
    Pdf,
    /// HTML format
    Html,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Excel format
    Excel,
    /// Custom format
    Custom(String),
}

impl ComplianceContext {
    /// Create a new compliance context
    pub fn new(id: String, config: ComplianceConfig) -> Self {
        Self {
            id,
            state: Arc::new(RwLock::new(ComplianceState::Initializing)),
            data_governance: Arc::new(RwLock::new(DataGovernanceManager::new())),
            regulatory_manager: Arc::new(RwLock::new(RegulatoryFrameworkManager::new())),
            auditor: Arc::new(Mutex::new(ComplianceAuditor::new())),
            policy_engine: Arc::new(RwLock::new(CompliancePolicyEngine::new())),
            consent_manager: Arc::new(RwLock::new(ConsentManager::new())),
            retention_manager: Arc::new(Mutex::new(DataRetentionManager::new())),
            breach_detector: Arc::new(Mutex::new(BreachDetector::new())),
            metrics: Arc::new(Mutex::new(ComplianceMetrics::default())),
            config: Arc::new(RwLock::new(config)),
            created_at: SystemTime::now(),
        }
    }

    /// Initialize the compliance context
    pub fn initialize(&self) -> ContextResult<()> {
        let mut state = self.state.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;

        if *state != ComplianceState::Initializing {
            return Err(ContextError::custom("invalid_state",
                format!("Cannot initialize compliance context in state: {}", state)));
        }

        *state = ComplianceState::Active;
        Ok(())
    }

    /// Get compliance state
    pub fn get_state(&self) -> ContextResult<ComplianceState> {
        let state = self.state.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;
        Ok(*state)
    }

    /// Get compliance metrics
    pub fn get_metrics(&self) -> ContextResult<ComplianceMetrics> {
        let metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        Ok(metrics.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_context_creation() {
        let config = ComplianceConfig::default();
        let context = ComplianceContext::new("test-compliance".to_string(), config);
        assert_eq!(context.id, "test-compliance");
    }

    #[test]
    fn test_regulatory_frameworks() {
        assert_eq!(RegulatoryFramework::Gdpr.to_string(), "GDPR");
        assert_eq!(RegulatoryFramework::Hipaa.to_string(), "HIPAA");
        assert_eq!(RegulatoryFramework::Custom("TEST".to_string()).to_string(), "CUSTOM-TEST");
    }

    #[test]
    fn test_compliance_states() {
        assert_eq!(ComplianceState::Active.to_string(), "active");
        assert_eq!(ComplianceState::NonCompliant.to_string(), "non_compliant");
    }

    #[test]
    fn test_compliance_config_default() {
        let config = ComplianceConfig::default();
        assert!(config.enabled_frameworks.contains(&RegulatoryFramework::Gdpr));
        assert!(config.data_governance_settings.enable_classification);
        assert!(config.audit_settings.enable_auditing);
    }

    #[test]
    fn test_sensitivity_levels() {
        assert_eq!(SensitivityLevel::Medium, SensitivityLevel::Medium);
        assert_ne!(SensitivityLevel::Low, SensitivityLevel::High);
    }

    #[test]
    fn test_report_formats() {
        assert_eq!(ReportFormat::Pdf, ReportFormat::Pdf);
        assert_ne!(ReportFormat::Json, ReportFormat::Html);
    }
}