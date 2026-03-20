//! # BackupSystem - Trait Implementations
//!
//! This module contains trait implementations for `BackupSystem`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BackupSystem {
    fn default() -> Self {
        Self {
            enabled: true,
            strategies: vec![BackupStrategy::Incremental],
            scheduling: BackupScheduling::default(),
            retention: BackupRetention::default(),
            verification: BackupVerification::default(),
        }
    }
}

