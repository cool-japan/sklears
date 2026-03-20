//! # ErrorClassification - Trait Implementations
//!
//! This module contains trait implementations for `ErrorClassification`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    DifficultyLevel, ErrorCategory, ErrorClassification, ErrorFrequency, SeverityLevel,
};

impl Default for ErrorClassification {
    fn default() -> Self {
        Self {
            category: ErrorCategory::Unknown,
            severity: SeverityLevel::Medium,
            frequency: ErrorFrequency::Occasional,
            resolution_difficulty: DifficultyLevel::Medium,
        }
    }
}
