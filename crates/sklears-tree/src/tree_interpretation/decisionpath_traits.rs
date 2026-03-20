//! # DecisionPath - Trait Implementations
//!
//! This module contains trait implementations for `DecisionPath`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::error::{Result, SklearsError};

use super::types::DecisionPath;

impl fmt::Display for DecisionPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.conditions.is_empty() {
            return write!(f, "Root -> Prediction: {}", self.prediction);
        }
        let conditions_str = self
            .conditions
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" AND ");
        write!(f, "IF {} THEN Prediction: {}", conditions_str, self.prediction)?;
        if let Some(confidence) = self.confidence {
            write!(f, " (confidence: {:.3})", confidence)?;
        }
        if let Some(n_samples) = self.n_samples {
            write!(f, " [samples: {}]", n_samples)?;
        }
        Ok(())
    }
}

