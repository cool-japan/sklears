//! # MemoryConfig - Trait Implementations
//!
//! This module contains trait implementations for `MemoryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory: Some(8 * 1024 * 1024 * 1024),
            limit_enforcement: MemoryLimitEnforcement::Strict,
            swap_policy: SwapPolicy::Auto,
            allocation_strategy: MemoryAllocationStrategy::FirstFit,
            huge_pages: HugePagesConfig::default(),
            protection: MemoryProtectionConfig::default(),
        }
    }
}

