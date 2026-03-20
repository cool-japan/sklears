//! # PluginConfig - Trait Implementations
//!
//! This module contains trait implementations for `PluginConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::path::PathBuf;

use super::types::PluginConfig;

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            plugin_dirs: vec![PathBuf::from("./plugins")],
            auto_load: true,
            sandbox: false,
            max_execution_time: std::time::Duration::from_secs(300),
            validate_plugins: true,
        }
    }
}
