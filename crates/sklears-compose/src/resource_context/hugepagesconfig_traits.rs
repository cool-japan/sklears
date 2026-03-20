//! # HugePagesConfig - Trait Implementations
//!
//! This module contains trait implementations for `HugePagesConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for HugePagesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            page_size: HugePageSize::Size2MB,
            page_count: None,
            allocation_policy: HugePagePolicy::Auto,
        }
    }
}

