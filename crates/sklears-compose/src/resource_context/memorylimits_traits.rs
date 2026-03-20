//! # MemoryLimits - Trait Implementations
//!
//! This module contains trait implementations for `MemoryLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_rss: Some(8 * 1024 * 1024 * 1024),
            max_vm: Some(16 * 1024 * 1024 * 1024),
            max_stack: Some(8 * 1024 * 1024),
            max_heap: Some(8 * 1024 * 1024 * 1024),
        }
    }
}

