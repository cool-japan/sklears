//! # NetworkIsolationConfig - Trait Implementations
//!
//! This module contains trait implementations for `NetworkIsolationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for NetworkIsolationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: None,
            vlan_id: None,
            firewall_rules: Vec::new(),
        }
    }
}

