//! # ExtensionStatus - Trait Implementations
//!
//! This module contains trait implementations for `ExtensionStatus`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ExtensionStatus;

impl Display for ExtensionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionStatus::Discovered => write!(f, "discovered"),
            ExtensionStatus::Loading => write!(f, "loading"),
            ExtensionStatus::Active => write!(f, "active"),
            ExtensionStatus::Paused => write!(f, "paused"),
            ExtensionStatus::Unloading => write!(f, "unloading"),
            ExtensionStatus::Unloaded => write!(f, "unloaded"),
            ExtensionStatus::Error => write!(f, "error"),
            ExtensionStatus::Disabled => write!(f, "disabled"),
        }
    }
}

