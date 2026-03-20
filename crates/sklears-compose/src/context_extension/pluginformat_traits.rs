//! # PluginFormat - Trait Implementations
//!
//! This module contains trait implementations for `PluginFormat`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PluginFormat;

impl Display for PluginFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginFormat::DynamicLibrary => write!(f, "dynamic_library"),
            PluginFormat::WebAssembly => write!(f, "webassembly"),
            PluginFormat::JavaScript => write!(f, "javascript"),
            PluginFormat::Python => write!(f, "python"),
            PluginFormat::Lua => write!(f, "lua"),
            PluginFormat::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

