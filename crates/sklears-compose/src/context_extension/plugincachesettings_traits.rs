//! # PluginCacheSettings - Trait Implementations
//!
//! This module contains trait implementations for `PluginCacheSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PluginCacheSettings;

impl Default for PluginCacheSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_dir: PathBuf::from("./cache/plugins"),
            ttl: Duration::from_secs(24 * 60 * 60),
            max_size: 1024 * 1024 * 1024,
        }
    }
}

