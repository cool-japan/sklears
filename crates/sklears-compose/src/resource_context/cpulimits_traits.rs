//! # CpuLimits - Trait Implementations
//!
//! This module contains trait implementations for `CpuLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for CpuLimits {
    fn default() -> Self {
        Self {
            max_cores: Some(num_cpus::get() as f32),
            max_usage_percent: Some(100.0),
            cpu_time_limit: None,
        }
    }
}

