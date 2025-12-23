//! # BackupRetention - Trait Implementations
//!
//! This module contains trait implementations for `BackupRetention`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BackupRetention {
    fn default() -> Self {
        let mut retention_periods = HashMap::new();
        retention_periods.insert("daily".to_string(), Duration::days(30));
        retention_periods.insert("weekly".to_string(), Duration::days(90));
        retention_periods.insert("monthly".to_string(), Duration::days(365));
        Self {
            retention_periods,
            archive_old_backups: true,
            max_backup_count: Some(100),
        }
    }
}

