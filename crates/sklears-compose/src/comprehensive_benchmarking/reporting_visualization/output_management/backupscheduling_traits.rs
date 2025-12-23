//! # BackupScheduling - Trait Implementations
//!
//! This module contains trait implementations for `BackupScheduling`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BackupScheduling {
    fn default() -> Self {
        Self {
            frequency: BackupFrequency::Daily,
            time_windows: vec!["02:00-04:00".to_string()],
            priorities: BackupPriorities::default(),
        }
    }
}

