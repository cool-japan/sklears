//! # PluginLoaderConfig - Trait Implementations
//!
//! This module contains trait implementations for `PluginLoaderConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PluginLoaderConfig;

impl Default for PluginLoaderConfig {
    fn default() -> Self {
        Self {
            parallel_loading: true,
            load_timeout: Duration::from_secs(30),
            enable_preloading: false,
            cache_validation_interval: Duration::from_secs(60 * 60),
        }
    }
}

