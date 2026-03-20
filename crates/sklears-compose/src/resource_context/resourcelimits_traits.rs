//! # ResourceLimits - Trait Implementations
//!
//! This module contains trait implementations for `ResourceLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu: CpuLimits::default(),
            memory: MemoryLimits::default(),
            storage: StorageLimits::default(),
            network: NetworkLimits::default(),
            gpu: None,
            custom: HashMap::new(),
        }
    }
}

