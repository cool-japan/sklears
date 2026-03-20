//! # BackupPriorities - Trait Implementations
//!
//! This module contains trait implementations for `BackupPriorities`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BackupPriorities {
    fn default() -> Self {
        Self {
            high_priority: vec!["*.important".to_string(), "critical/*".to_string()],
            medium_priority: vec!["*.data".to_string()],
            low_priority: vec!["*.temp".to_string(), "cache/*".to_string()],
        }
    }
}

