//! # CpuThrottling - Trait Implementations
//!
//! This module contains trait implementations for `CpuThrottling`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for CpuThrottling {
    fn default() -> Self {
        Self {
            enabled: false,
            max_usage_percent: 100.0,
            period: Duration::from_millis(100),
            quota: Duration::from_millis(100),
        }
    }
}

