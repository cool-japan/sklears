//! # OvercommitPolicies - Trait Implementations
//!
//! This module contains trait implementations for `OvercommitPolicies`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for OvercommitPolicies {
    fn default() -> Self {
        Self {
            cpu_overcommit_ratio: 2.0,
            memory_overcommit_ratio: 1.5,
            storage_overcommit_ratio: 3.0,
            network_overcommit_ratio: 2.0,
            dynamic_adjustment: true,
        }
    }
}

