//! # AccessControlManager - Trait Implementations
//!
//! This module contains trait implementations for `AccessControlManager`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for AccessControlManager {
    fn default() -> Self {
        Self {
            enabled: false,
            policies: vec![],
            user_groups: HashMap::new(),
            audit_logging: AuditLogging::default(),
        }
    }
}

