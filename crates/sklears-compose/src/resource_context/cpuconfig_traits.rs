//! # CpuConfig - Trait Implementations
//!
//! This module contains trait implementations for `CpuConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            max_cores: Some(num_cpus::get()),
            affinity_mask: None,
            priority: CpuPriority::Normal,
            throttling: CpuThrottling::default(),
            numa_preference: NumaPreference::Any,
        }
    }
}

