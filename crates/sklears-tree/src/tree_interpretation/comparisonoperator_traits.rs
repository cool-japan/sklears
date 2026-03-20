//! # ComparisonOperator - Trait Implementations
//!
//! This module contains trait implementations for `ComparisonOperator`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::error::{Result, SklearsError};

use super::types::ComparisonOperator;

impl fmt::Display for ComparisonOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparisonOperator::LessThanOrEqual => write!(f, "<="),
            ComparisonOperator::GreaterThan => write!(f, ">"),
            ComparisonOperator::Equal => write!(f, "=="),
            ComparisonOperator::NotEqual => write!(f, "!="),
            ComparisonOperator::In => write!(f, "in"),
            ComparisonOperator::NotIn => write!(f, "not in"),
        }
    }
}

