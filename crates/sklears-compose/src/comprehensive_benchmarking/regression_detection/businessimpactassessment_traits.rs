//! # `BusinessImpactAssessment` - Trait Implementations
//!
//! This module contains trait implementations for `BusinessImpactAssessment`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BusinessImpactAssessment;

impl Default for BusinessImpactAssessment {
    fn default() -> Self {
        Self {
            revenue_impact: 0.0,
            customer_impact: 0.0,
            operational_impact: 0.0,
            reputation_impact: 0.0,
        }
    }
}
