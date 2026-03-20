//! # StorageLimits - Trait Implementations
//!
//! This module contains trait implementations for `StorageLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for StorageLimits {
    fn default() -> Self {
        Self {
            max_space: Some(100 * 1024 * 1024 * 1024),
            max_files: Some(1000000),
            max_iops: Some(10000),
            max_bandwidth: Some(1024 * 1024 * 1024),
        }
    }
}

