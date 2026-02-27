//! # ExtensionType - Trait Implementations
//!
//! This module contains trait implementations for `ExtensionType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ExtensionType;

impl Display for ExtensionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionType::Plugin => write!(f, "plugin"),
            ExtensionType::Middleware => write!(f, "middleware"),
            ExtensionType::Filter => write!(f, "filter"),
            ExtensionType::Processor => write!(f, "processor"),
            ExtensionType::Renderer => write!(f, "renderer"),
            ExtensionType::Validator => write!(f, "validator"),
            ExtensionType::Transformer => write!(f, "transformer"),
            ExtensionType::Handler => write!(f, "handler"),
            ExtensionType::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

