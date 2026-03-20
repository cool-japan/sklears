//! # PermissionLevel - Trait Implementations
//!
//! This module contains trait implementations for `PermissionLevel`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PermissionLevel;

impl Display for PermissionLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PermissionLevel::None => write!(f, "none"),
            PermissionLevel::Restricted => write!(f, "restricted"),
            PermissionLevel::Normal => write!(f, "normal"),
            PermissionLevel::Elevated => write!(f, "elevated"),
            PermissionLevel::Administrative => write!(f, "administrative"),
            PermissionLevel::Full => write!(f, "full"),
        }
    }
}

