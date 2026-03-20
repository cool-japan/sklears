//! # StorageConfig - Trait Implementations
//!
//! This module contains trait implementations for `StorageConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_storage: Some(100 * 1024 * 1024 * 1024),
            temp_storage_limit: Some(10 * 1024 * 1024 * 1024),
            iops_limit: Some(10000),
            bandwidth_limit: Some(1024 * 1024 * 1024),
            priority: StoragePriority::Normal,
            fs_quotas: HashMap::new(),
            encryption: StorageEncryptionConfig::default(),
        }
    }
}

