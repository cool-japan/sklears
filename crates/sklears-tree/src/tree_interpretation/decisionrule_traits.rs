//! # DecisionRule - Trait Implementations
//!
//! This module contains trait implementations for `DecisionRule`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::error::{Result, SklearsError};

use super::types::DecisionRule;

impl fmt::Display for DecisionRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.antecedent.is_empty() {
            return write!(f, "PREDICT {}", self.consequent);
        }
        let conditions_str = self
            .antecedent
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" AND ");
        write!(
            f, "IF {} THEN PREDICT {} (support: {:.3}, confidence: {:.3}",
            conditions_str, self.consequent, self.support, self.confidence
        )?;
        if let Some(lift) = self.lift {
            write!(f, ", lift: {:.3}", lift)?;
        }
        write!(f, ")")
    }
}

