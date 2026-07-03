//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

use super::types::ComplianceAssessmentResult;

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
