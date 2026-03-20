//! # DecisionCondition - Trait Implementations
//!
//! This module contains trait implementations for `DecisionCondition`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::error::{Result, SklearsError};

use super::types::DecisionCondition;

impl fmt::Display for DecisionCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let default_name = format!("feature_{}", self.feature_idx);
        let feature_name = self
            .feature_name
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or(&default_name);
        write!(f, "{} {} {}", feature_name, self.operator, self.threshold)
    }
}

