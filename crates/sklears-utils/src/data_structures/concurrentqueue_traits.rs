//! # ConcurrentQueue - Trait Implementations
//!
//! This module contains trait implementations for `ConcurrentQueue`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ConcurrentQueue;

impl<T: Clone> Default for ConcurrentQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}
