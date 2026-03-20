//! # ResourceManagerConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourceManagerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceManagerConfig;

impl Default for ResourceManagerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(10),
            enable_pooling: true,
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

