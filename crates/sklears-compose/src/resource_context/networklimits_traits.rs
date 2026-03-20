//! # NetworkLimits - Trait Implementations
//!
//! This module contains trait implementations for `NetworkLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for NetworkLimits {
    fn default() -> Self {
        Self {
            max_bandwidth: Some(1024 * 1024 * 1024),
            max_connections: Some(10000),
            max_pps: Some(100000),
        }
    }
}

