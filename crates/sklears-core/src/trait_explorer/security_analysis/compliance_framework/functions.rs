//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::security_types::*;
use std::time::{Duration, SystemTime};

use super::macros::{
    AlertLevel, AlertThreshold, AutomatedComplianceTesting, ComplianceMetrics,
    ContinuousComplianceMonitoring, MetricDefinition, MonitoringRule, TestCase, TestSchedule,
    TestSuite,
};
use super::types::ComplianceAssessmentResult;
use super::types_5::ComplianceFrameworkManager;
use super::types_6::{ComplianceError, ComplianceStatus};

/// Shared `items_met`/`items_total` -> [`ComplianceStatus`] reduction used by both the
/// regulatory-framework and security-standard coverage assessments above.
pub(super) fn compliance_status_from_coverage(
    items_met: u32,
    items_total: u32,
) -> ComplianceStatus {
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
/// Rank a [`RiskSeverity`] for sorting purposes (higher rank = more severe). `RiskSeverity` does
/// not derive `Ord` (it intentionally has no single global "severity order" semantics elsewhere
/// in the security-analysis framework), so callers that need a total order build one locally.
pub(super) fn risk_severity_rank(severity: &RiskSeverity) -> u8 {
    match severity {
        RiskSeverity::Low => 1,
        RiskSeverity::Medium => 2,
        RiskSeverity::High => 3,
        RiskSeverity::Critical => 4,
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
