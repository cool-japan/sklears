//! # ExtensionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ExtensionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ExtensionConfig, ExtensionResourceLimits, ExtensionSecuritySettings, PluginSettings};

impl Default for ExtensionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            extensions_dir: PathBuf::from("./extensions"),
            enable_sandboxing: true,
            enable_hot_reload: false,
            extension_timeout: Duration::from_secs(30),
            max_concurrent_extensions: 100,
            security_settings: ExtensionSecuritySettings::default(),
            resource_limits: ExtensionResourceLimits::default(),
            plugin_settings: PluginSettings::default(),
            custom: HashMap::new(),
        }
    }
}

