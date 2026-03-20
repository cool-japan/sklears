//! # MemoryProtectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `MemoryProtectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for MemoryProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            stack_guard: true,
            heap_guard: true,
            aslr: true,
            dep: true,
        }
    }
}

