//! # CleanupPolicy - Trait Implementations
//!
//! This module contains trait implementations for `CleanupPolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for CleanupPolicy {
    fn default() -> Self {
        Self {
            auto_cleanup: true,
            retention_period: Duration::days(30),
            size_limit: 10 * 1024 * 1024 * 1024,
            cleanup_frequency: Duration::hours(24),
            cleanup_triggers: vec![
                CleanupTrigger::Age(Duration::days(30)), CleanupTrigger::SizeLimit(0.9),
                CleanupTrigger::CountLimit(10000),
            ],
            cleanup_actions: vec![
                CleanupAction::Archive, CleanupAction::Compress, CleanupAction::Delete,
            ],
        }
    }
}

