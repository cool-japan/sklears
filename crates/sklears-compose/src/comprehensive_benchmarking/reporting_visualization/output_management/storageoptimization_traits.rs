//! # StorageOptimization - Trait Implementations
//!
//! This module contains trait implementations for `StorageOptimization`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for StorageOptimization {
    fn default() -> Self {
        Self {
            deduplication: true,
            compression: true,
            thin_provisioning: true,
            tiering_policies: vec![],
        }
    }
}

