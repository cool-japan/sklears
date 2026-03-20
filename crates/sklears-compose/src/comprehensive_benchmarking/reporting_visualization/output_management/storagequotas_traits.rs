//! # StorageQuotas - Trait Implementations
//!
//! This module contains trait implementations for `StorageQuotas`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for StorageQuotas {
    fn default() -> Self {
        Self {
            enabled: false,
            user_quotas: HashMap::new(),
            project_quotas: HashMap::new(),
            global_quota: None,
        }
    }
}

