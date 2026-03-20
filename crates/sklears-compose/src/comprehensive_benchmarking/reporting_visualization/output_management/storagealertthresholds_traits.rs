//! # StorageAlertThresholds - Trait Implementations
//!
//! This module contains trait implementations for `StorageAlertThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for StorageAlertThresholds {
    fn default() -> Self {
        Self {
            usage_warning: 0.8,
            usage_critical: 0.95,
            performance_threshold: 0.7,
        }
    }
}

