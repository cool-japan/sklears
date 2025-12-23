//! # BackupSettings - Trait Implementations
//!
//! This module contains trait implementations for `BackupSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BackupSettings {
    fn default() -> Self {
        Self {
            backup_enabled: false,
            backup_frequency: Duration::from_secs(24 * 3600),
            backup_location: "/tmp/metrics_backup".to_string(),
            max_backups: 7,
            compression_enabled: true,
        }
    }
}

