//! # `UserImpactAssessment` - Trait Implementations
//!
//! This module contains trait implementations for `UserImpactAssessment`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_20::UserImpactAssessment;

impl Default for UserImpactAssessment {
    fn default() -> Self {
        Self {
            affected_users: 0,
            user_experience_degradation: 0.0,
            feature_availability: 100.0,
            performance_perception: 0.0,
        }
    }
}
