//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::types_6::ComplianceFrameworkType;

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
