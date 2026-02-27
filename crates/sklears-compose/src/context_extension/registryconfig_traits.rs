//! # RegistryConfig - Trait Implementations
//!
//! This module contains trait implementations for `RegistryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RegistryConfig;

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_registry: "https://registry.extensions.com".to_string(),
            cache_ttl: Duration::from_secs(60 * 60),
            auto_update_interval: Duration::from_secs(24 * 60 * 60),
            enable_prerelease: false,
        }
    }
}

