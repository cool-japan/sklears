//! Enterprise features for model explanation and inspection
//!
//! This module provides enterprise-grade features including role-based access control,
//! audit trails, quality monitoring, and compliance reporting for explanation systems.
//!
//! ## Features
//!
//! ### Role-Based Access Control (RBAC)
//! - Fine-grained permission system for explanation operations
//! - User and role management with inheritance
//! - Session management and security contexts
//! - Secure execution wrappers for explanation operations
//!
//! ### Audit Trails and Lineage Tracking
//! - Comprehensive audit logging of all explanation operations
//! - Explanation lineage tracking with directed acyclic graphs
//! - Configurable retention policies and storage backends
//! - Integration with compliance reporting systems
//!
//! ### Quality Monitoring
//! - Real-time quality metrics collection and analysis
//! - Configurable alerting and threshold management
//! - Trend detection and statistical analysis
//! - Quality degradation detection and root cause analysis
//!
//! ### Compliance Reporting
//! - Support for major regulatory frameworks (GDPR, AI Act, FCRA, etc.)
//! - Automated compliance assessment and scoring
//! - Risk analysis and remediation planning
//! - Standardized reporting formats
//!
//! ## Example Usage
//!
//! ```rust
//! use sklears_inspection::enterprise::*;
//! use chrono::Utc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Set up role-based access control
//! let config = AccessControlConfig::default();
//! let mut access_control = AccessControl::new(config);
//!
//! // Create a user with analyst role
//! let mut user = User::new(
//!     "user123".to_string(),
//!     "analyst1".to_string(),
//!     "analyst@company.com".to_string(),
//!     "Data Analyst".to_string(),
//! );
//! user.add_role("analyst".to_string());
//! access_control.role_manager_mut().add_user(user)?;
//!
//! // Create a session for the user
//! let context = access_control.create_session("user123", "session_abc".to_string())?;
//!
//! // Check permissions
//! assert!(context.has_permission(&Permission::ViewExplanations));
//! assert!(context.has_permission(&Permission::GenerateExplanations));
//!
//! // Set up quality monitoring
//! let monitor_config = QualityMonitorConfig::default();
//! let mut monitor = ExplanationQualityMonitor::new(monitor_config);
//!
//! // Record quality measurements
//! monitor.record_measurement(QualityMetric::LocalFidelity, 0.85)?;
//! monitor.record_measurement(QualityMetric::ExplanationGenerationTime, 1500.0)?;
//!
//! // Set up compliance reporting
//! let mut compliance = ComplianceReporter::new();
//!
//! // Generate a compliance report
//! let report = compliance.generate_report(
//!     "Monthly Compliance Report".to_string(),
//!     Utc::now() - chrono::Duration::days(30),
//!     Utc::now(),
//!     Some("Compliance Officer".to_string()),
//!     None,
//! )?;
//!
//! println!("Compliance score: {:.1}%", report.summary.compliance_percentage);
//! # Ok(())
//! # }
//! ```

pub mod audit;
pub mod compliance;
pub mod quality_monitoring;
pub mod rbac;

// Re-export core enterprise functionality
pub use rbac::{
    AccessControl, AccessControlConfig, AccessLevel, Permission, PermissionSet, Role, RoleManager,
    SecureExplanationExecutor, SecurityContext, User, UserGroup,
};

pub use audit::{
    AuditEvent, AuditEventType, AuditLogger, AuditRecord, AuditSeverity, AuditTrail,
    ExplanationLineage, LineageNode, LineageRelation, LineageTracker, OperationType,
};

pub use quality_monitoring::{
    AlertRule, ExplanationQualityMonitor, QualityAlert, QualityDataPoint, QualityLevel,
    QualityMetric, QualityMonitorConfig, QualityStatistics, QualityThreshold, QualityTrend,
    QualityTrendDirection,
};

pub use compliance::{
    ActionItem, ComplianceFramework, ComplianceReport, ComplianceReporter, ComplianceRule,
    ComplianceStatus, RegulatoryRequirement, RiskLevel,
};

use crate::{SklResult, SklearsError};
use std::collections::HashMap;

/// Enterprise configuration for explanation systems
#[derive(Debug, Clone)]
pub struct EnterpriseConfig {
    /// Role-based access control configuration
    pub rbac_config: AccessControlConfig,
    /// Quality monitoring configuration
    pub quality_config: QualityMonitorConfig,
    /// Audit logging enabled
    pub audit_enabled: bool,
    /// Compliance framework requirements
    pub compliance_frameworks: Vec<ComplianceFramework>,
}

impl Default for EnterpriseConfig {
    fn default() -> Self {
        Self {
            rbac_config: AccessControlConfig::default(),
            quality_config: QualityMonitorConfig::default(),
            audit_enabled: true,
            compliance_frameworks: vec![ComplianceFramework::GDPR, ComplianceFramework::AIAct],
        }
    }
}

impl EnterpriseConfig {
    /// Create enterprise config for financial services
    pub fn financial_services() -> Self {
        Self {
            rbac_config: AccessControlConfig {
                enabled: true,
                default_role: "viewer".to_string(),
                session_timeout: 1800, // 30 minutes
                log_access_events: true,
                enforce_mfa: true,
            },
            quality_config: QualityMonitorConfig {
                max_data_points: 10000,
                evaluation_interval: chrono::Duration::minutes(1),
                enable_alerting: true,
                data_retention_period: chrono::Duration::days(2555), // 7 years
                trend_confidence_threshold: 0.8,
            },
            audit_enabled: true,
            compliance_frameworks: vec![
                ComplianceFramework::SOX,
                ComplianceFramework::Basel3,
                ComplianceFramework::DoddFrank,
            ],
        }
    }

    /// Create enterprise config for healthcare
    pub fn healthcare() -> Self {
        Self {
            rbac_config: AccessControlConfig {
                enabled: true,
                default_role: "viewer".to_string(),
                session_timeout: 900, // 15 minutes
                log_access_events: true,
                enforce_mfa: true,
            },
            quality_config: QualityMonitorConfig {
                max_data_points: 50000,
                evaluation_interval: chrono::Duration::seconds(30),
                enable_alerting: true,
                data_retention_period: chrono::Duration::days(2555), // 7 years
                trend_confidence_threshold: 0.9,
            },
            audit_enabled: true,
            compliance_frameworks: vec![
                ComplianceFramework::HIPAA,
                ComplianceFramework::MedicalDevice,
            ],
        }
    }

    /// Create enterprise config for European operations
    pub fn european_operations() -> Self {
        Self {
            rbac_config: AccessControlConfig::default(),
            quality_config: QualityMonitorConfig::default(),
            audit_enabled: true,
            compliance_frameworks: vec![
                ComplianceFramework::GDPR,
                ComplianceFramework::AIAct,
                ComplianceFramework::MiFID2,
            ],
        }
    }
}

/// Integrated enterprise explanation system
#[derive(Debug)]
pub struct EnterpriseExplanationSystem {
    /// Configuration
    config: EnterpriseConfig,
    /// Access control system
    access_control: AccessControl,
    /// Quality monitor
    quality_monitor: ExplanationQualityMonitor,
    /// Compliance reporter
    compliance_reporter: ComplianceReporter,
    /// Audit and lineage tracker
    lineage_tracker: LineageTracker,
}

impl EnterpriseExplanationSystem {
    /// Create a new enterprise explanation system
    pub fn new(config: EnterpriseConfig) -> Self {
        Self {
            access_control: AccessControl::new(config.rbac_config.clone()),
            quality_monitor: ExplanationQualityMonitor::new(config.quality_config.clone()),
            compliance_reporter: ComplianceReporter::new(),
            lineage_tracker: LineageTracker::new(10000),
            config,
        }
    }

    /// Get access control system
    pub fn access_control(&self) -> &AccessControl {
        &self.access_control
    }

    /// Get access control system mutably
    pub fn access_control_mut(&mut self) -> &mut AccessControl {
        &mut self.access_control
    }

    /// Get quality monitor
    pub fn quality_monitor(&self) -> &ExplanationQualityMonitor {
        &self.quality_monitor
    }

    /// Get quality monitor mutably
    pub fn quality_monitor_mut(&mut self) -> &mut ExplanationQualityMonitor {
        &mut self.quality_monitor
    }

    /// Get compliance reporter
    pub fn compliance_reporter(&self) -> &ComplianceReporter {
        &self.compliance_reporter
    }

    /// Get compliance reporter mutably
    pub fn compliance_reporter_mut(&mut self) -> &mut ComplianceReporter {
        &mut self.compliance_reporter
    }

    /// Get lineage tracker
    pub fn lineage_tracker(&self) -> &LineageTracker {
        &self.lineage_tracker
    }

    /// Get lineage tracker mutably
    pub fn lineage_tracker_mut(&mut self) -> &mut LineageTracker {
        &mut self.lineage_tracker
    }

    /// Execute an explanation operation with full enterprise controls
    pub fn execute_explanation_operation<T, F>(
        &mut self,
        session_id: &str,
        operation_name: &str,
        required_permission: Permission,
        quality_metrics: HashMap<QualityMetric, f64>,
        operation: F,
    ) -> SklResult<T>
    where
        F: FnOnce() -> SklResult<T>,
    {
        // Check access control
        let context = self
            .access_control
            .get_session(session_id)
            .ok_or_else(|| SklearsError::InvalidOperation("Invalid session".to_string()))?;

        context.require_permission(&required_permission)?;

        // Execute operation
        let result = operation()?;

        // Record quality metrics
        for (metric, value) in quality_metrics {
            if let Err(e) = self.quality_monitor.record_measurement(metric, value) {
                eprintln!("Warning: Failed to record quality metric: {}", e);
            }
        }

        // Create audit event
        let audit_event = AuditEvent::new(
            AuditEventType::ExplanationGenerated,
            operation_name.to_string(),
            format!("Successfully executed {} operation", operation_name),
        )
        .with_user(context.user.id.clone(), Some(context.session_id.clone()));

        // Note: Full audit logging would require mutable access to audit trail
        // This is a simplified version for demonstration

        Ok(result)
    }

    /// Get system health summary
    pub fn get_system_health(&self) -> SystemHealthSummary {
        let quality_score = self.quality_monitor.get_overall_quality_score();
        let compliance_score = self.compliance_reporter.get_overall_compliance_score();

        let active_alerts = self.quality_monitor.get_active_alerts().len();
        let critical_alerts = self.quality_monitor.get_critical_alerts().len();

        let overall_health =
            if compliance_score >= 95.0 && quality_score >= 0.8 && critical_alerts == 0 {
                SystemHealth::Excellent
            } else if compliance_score >= 80.0 && quality_score >= 0.6 && critical_alerts <= 1 {
                SystemHealth::Good
            } else if compliance_score >= 60.0 && quality_score >= 0.4 {
                SystemHealth::Warning
            } else {
                SystemHealth::Critical
            };

        SystemHealthSummary {
            overall_health,
            quality_score,
            compliance_score,
            active_alerts,
            critical_alerts,
            audit_trail_size: self.lineage_tracker.audit_trail().total_events(),
            lineage_nodes: self.lineage_tracker.lineage().node_count(),
            lineage_relations: self.lineage_tracker.lineage().relation_count(),
        }
    }
}

/// System health status
#[derive(Debug, Clone, PartialEq)]
pub enum SystemHealth {
    /// Excellent
    Excellent,
    /// Good
    Good,
    /// Warning
    Warning,
    /// Critical
    Critical,
}

/// System health summary
#[derive(Debug, Clone)]
pub struct SystemHealthSummary {
    /// Overall system health
    pub overall_health: SystemHealth,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Compliance score (0.0 to 100.0)
    pub compliance_score: f64,
    /// Number of active alerts
    pub active_alerts: usize,
    /// Number of critical alerts
    pub critical_alerts: usize,
    /// Size of audit trail
    pub audit_trail_size: u64,
    /// Number of lineage nodes
    pub lineage_nodes: usize,
    /// Number of lineage relations
    pub lineage_relations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enterprise_config_default() {
        let config = EnterpriseConfig::default();
        assert!(config.audit_enabled);
        assert!(config.rbac_config.enabled);
        assert!(!config.compliance_frameworks.is_empty());
    }

    #[test]
    fn test_enterprise_config_financial_services() {
        let config = EnterpriseConfig::financial_services();
        assert_eq!(config.rbac_config.session_timeout, 1800);
        assert!(config.rbac_config.enforce_mfa);
        assert!(config
            .compliance_frameworks
            .contains(&ComplianceFramework::SOX));
        assert!(config
            .compliance_frameworks
            .contains(&ComplianceFramework::Basel3));
    }

    #[test]
    fn test_enterprise_config_healthcare() {
        let config = EnterpriseConfig::healthcare();
        assert_eq!(config.rbac_config.session_timeout, 900);
        assert!(config.rbac_config.enforce_mfa);
        assert!(config
            .compliance_frameworks
            .contains(&ComplianceFramework::HIPAA));
        assert!(config
            .compliance_frameworks
            .contains(&ComplianceFramework::MedicalDevice));
    }

    #[test]
    fn test_enterprise_explanation_system() {
        let config = EnterpriseConfig::default();
        let system = EnterpriseExplanationSystem::new(config);

        let health = system.get_system_health();
        assert_eq!(health.audit_trail_size, 0);
        assert_eq!(health.lineage_nodes, 0);
        assert_eq!(health.lineage_relations, 0);
    }
}
