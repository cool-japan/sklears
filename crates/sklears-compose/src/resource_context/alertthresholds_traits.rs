//! # AlertThresholds - Trait Implementations
//!
//! This module contains trait implementations for `AlertThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage: 80.0,
            memory_usage: 85.0,
            storage_usage: 90.0,
            network_usage: 80.0,
            gpu_usage: Some(80.0),
        }
    }
}

