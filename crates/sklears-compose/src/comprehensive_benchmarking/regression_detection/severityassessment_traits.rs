//! # `SeverityAssessment` - Trait Implementations
//!
//! This module contains trait implementations for `SeverityAssessment`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{BusinessImpactAssessment, OverallRegressionSeverity, SeverityAssessment};
use super::types_20::UserImpactAssessment;

impl Default for SeverityAssessment {
    fn default() -> Self {
        Self {
            overall_severity: OverallRegressionSeverity::Low,
            critical_regressions: 0,
            high_severity_regressions: 0,
            medium_severity_regressions: 0,
            low_severity_regressions: 0,
            total_regressions: 0,
            risk_score: 0.0,
            business_impact: BusinessImpactAssessment::default(),
            user_impact: UserImpactAssessment::default(),
            estimated_recovery_time: Duration::from_secs(0),
        }
    }
}
