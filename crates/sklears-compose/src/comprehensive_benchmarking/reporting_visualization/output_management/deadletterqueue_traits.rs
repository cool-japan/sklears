//! # DeadLetterQueue - Trait Implementations
//!
//! This module contains trait implementations for `DeadLetterQueue`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for DeadLetterQueue {
    fn default() -> Self {
        Self {
            enabled: true,
            storage_type: DlqStorageType::Database,
            retention_period: Duration::days(7),
            processing_options: DlqProcessingOptions::default(),
        }
    }
}

