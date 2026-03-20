//! # ExtensionState - Trait Implementations
//!
//! This module contains trait implementations for `ExtensionState`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ExtensionState;

impl Display for ExtensionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionState::Initializing => write!(f, "initializing"),
            ExtensionState::Active => write!(f, "active"),
            ExtensionState::Loading => write!(f, "loading"),
            ExtensionState::SafeMode => write!(f, "safe_mode"),
            ExtensionState::Disabled => write!(f, "disabled"),
            ExtensionState::Maintenance => write!(f, "maintenance"),
            ExtensionState::ShuttingDown => write!(f, "shutting_down"),
        }
    }
}

