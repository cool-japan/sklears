//! # `ComplianceConfiguration` - Trait Implementations
//!
//! This module contains trait implementations for `ComplianceConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types_6::ComplianceFrameworkType;
use super::types_7::ComplianceConfiguration;

impl Default for ComplianceConfiguration {
    fn default() -> Self {
        let mut assessment_frequency = HashMap::new();
        assessment_frequency.insert("GDPR".to_string(), Duration::from_secs(86400 * 30));
        assessment_frequency.insert("HIPAA".to_string(), Duration::from_secs(86400 * 90));
        assessment_frequency.insert("SOC2".to_string(), Duration::from_secs(86400 * 365));
        Self {
            enabled_frameworks: vec![
                ComplianceFrameworkType::NIST,
                ComplianceFrameworkType::GDPR,
                ComplianceFrameworkType::HIPAA,
                ComplianceFrameworkType::SOC2,
            ],
            assessment_frequency,
            audit_retention_period: Duration::from_secs(86400 * 365 * 7),
            automated_monitoring: true,
            real_time_alerting: true,
            compliance_threshold: 0.85,
            evidence_collection_enabled: true,
            continuous_assessment: true,
        }
    }
}
